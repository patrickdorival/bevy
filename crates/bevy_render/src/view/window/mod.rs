use crate::renderer::WgpuWrapper;
use crate::{
    camera::ExtractedCamera,
    render_resource::{SurfaceTexture, TextureView},
    renderer::{RenderAdapter, RenderDevice, RenderInstance},
    Extract, ExtractSchedule, Render, RenderApp, RenderSystems,
};
use bevy_app::{App, Plugin};
use bevy_camera::NormalizedRenderTarget;
use bevy_ecs::{entity::EntityHashMap, prelude::*};
use bevy_platform::collections::HashSet;
use bevy_utils::default;
use bevy_window::{
    CompositeAlphaMode, PresentMode, PrimaryWindow, RawHandleWrapper, Window, WindowClosing,
};
use core::{
    num::NonZero,
    ops::{Deref, DerefMut},
};
use tracing::{debug, info, trace, warn};
use wgpu::{
    SurfaceConfiguration, SurfaceTargetUnsafe, TextureFormat, TextureUsages, TextureViewDescriptor,
};

pub mod screenshot;

use screenshot::ScreenshotPlugin;

pub struct WindowRenderPlugin;

impl Plugin for WindowRenderPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ScreenshotPlugin);

        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .init_resource::<ExtractedWindows>()
                .init_resource::<WindowSurfaces>()
                .add_systems(ExtractSchedule, extract_windows)
                .add_systems(
                    Render,
                    create_surfaces
                        .run_if(need_surface_configuration)
                        .before(prepare_windows),
                )
                .add_systems(Render, prepare_windows.in_set(RenderSystems::ManageViews))
                .add_systems(
                    Render,
                    blit_offscreen_to_window.in_set(RenderSystems::Cleanup),
                );

            #[cfg(target_os = "macos")]
            render_app.add_systems(
                Render,
                init_metal_presenter
                    .in_set(RenderSystems::ManageViews)
                    .before(prepare_windows),
            );
        }
    }

    fn finish(&self, app: &mut App) {
        let Some(config) = app.world().get_resource::<OffscreenPresentConfig>().cloned() else {
            return;
        };

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        let render_device = render_app.world().resource::<RenderDevice>();

        let texture = render_device.wgpu_device().create_texture(&wgpu::TextureDescriptor {
            label: Some("offscreen_render_target"),
            size: wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TextureFormat::Bgra8UnormSrgb,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let view = texture.create_view(&TextureViewDescriptor::default());
        let bevy_view = TextureView::from(view);
        let manual_view = crate::texture::ManualTextureView {
            texture_view: bevy_view,
            size: bevy_math::UVec2::new(config.width, config.height),
            view_format: TextureFormat::Bgra8UnormSrgb,
            scale_factor: config.scale_factor,
        };

        // Insert into render world (init if not yet created by extraction)
        if !render_app.world().contains_resource::<crate::texture::ManualTextureViews>() {
            render_app.init_resource::<crate::texture::ManualTextureViews>();
        }
        render_app.world_mut().resource_mut::<crate::texture::ManualTextureViews>()
            .insert(config.handle, manual_view.clone());
        render_app.insert_resource(OffscreenBlitState {
            source_texture: texture,
            #[cfg(target_os = "macos")]
            metal_state: None,
            shutting_down: alloc::sync::Arc::new(core::sync::atomic::AtomicBool::new(false)),
        });

        // Share the shutdown flag with the main world so it can signal on exit.
        let shutdown_flag = render_app.world().resource::<OffscreenBlitState>().shutting_down.clone();
        app.insert_resource(OffscreenShutdownFlag(shutdown_flag));
        app.add_systems(bevy_app::Last, signal_offscreen_shutdown);

        // Insert into main world so camera_system can resolve target info
        app.world_mut().resource_mut::<crate::texture::ManualTextureViews>()
            .insert(config.handle, manual_view);

        info!(
            "Offscreen present: {}x{} Bgra8UnormSrgb (handle {:?})",
            config.width, config.height, config.handle
        );
    }
}

/// Main-world resource carrying the shutdown flag shared with the render world.
#[derive(Resource)]
pub struct OffscreenShutdownFlag(pub alloc::sync::Arc<core::sync::atomic::AtomicBool>);

/// Set the shutdown flag when the app is about to exit so the render-thread
/// blit doesn't block on nextDrawable for a destroyed window.
fn signal_offscreen_shutdown(
    mut exit_events: bevy_ecs::message::MessageReader<bevy_app::AppExit>,
    flag: Option<Res<OffscreenShutdownFlag>>,
) {
    if exit_events.read().next().is_some() {
        if let Some(flag) = flag {
            flag.0.store(true, core::sync::atomic::Ordering::Relaxed);
        }
    }
}

pub struct ExtractedWindow {
    /// An entity that contains the components in [`Window`].
    pub entity: Entity,
    pub handle: RawHandleWrapper,
    pub physical_width: u32,
    pub physical_height: u32,
    pub present_mode: PresentMode,
    pub desired_maximum_frame_latency: Option<NonZero<u32>>,
    /// Note: this will not always be the swap chain texture view. When taking a screenshot,
    /// this will point to an alternative texture instead to allow for copying the render result
    /// to CPU memory.
    pub swap_chain_texture_view: Option<TextureView>,
    pub swap_chain_texture: Option<SurfaceTexture>,
    pub swap_chain_texture_format: Option<TextureFormat>,
    pub swap_chain_texture_view_format: Option<TextureFormat>,
    pub size_changed: bool,
    pub present_mode_changed: bool,
    pub alpha_mode: CompositeAlphaMode,
    /// Whether this window needs an initial buffer commit.
    ///
    /// On Wayland, windows must present at least once before they are shown.
    /// See <https://wayland.app/protocols/xdg-shell#xdg_surface>
    pub needs_initial_present: bool,
}

impl ExtractedWindow {
    fn set_swapchain_texture(&mut self, frame: wgpu::SurfaceTexture) {
        self.swap_chain_texture_view_format = Some(frame.texture.format().add_srgb_suffix());
        let texture_view_descriptor = TextureViewDescriptor {
            format: self.swap_chain_texture_view_format,
            ..default()
        };
        self.swap_chain_texture_view = Some(TextureView::from(
            frame.texture.create_view(&texture_view_descriptor),
        ));
        self.swap_chain_texture = Some(SurfaceTexture::from(frame));
    }

    fn has_swapchain_texture(&self) -> bool {
        self.swap_chain_texture_view.is_some() && self.swap_chain_texture.is_some()
    }

    pub fn present(&mut self) {
        if let Some(surface_texture) = self.swap_chain_texture.take() {
            // TODO(clean): winit docs recommends calling pre_present_notify before this.
            // though `present()` doesn't present the frame, it schedules it to be presented
            // by wgpu.
            // https://docs.rs/winit/0.29.9/wasm32-unknown-unknown/winit/window/struct.Window.html#method.pre_present_notify
            surface_texture.present();
        }
    }
}

#[derive(Default, Resource)]
pub struct ExtractedWindows {
    pub primary: Option<Entity>,
    pub windows: EntityHashMap<ExtractedWindow>,
}

impl Deref for ExtractedWindows {
    type Target = EntityHashMap<ExtractedWindow>;

    fn deref(&self) -> &Self::Target {
        &self.windows
    }
}

impl DerefMut for ExtractedWindows {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.windows
    }
}

fn extract_windows(
    mut extracted_windows: ResMut<ExtractedWindows>,
    mut closing: Extract<MessageReader<WindowClosing>>,
    windows: Extract<Query<(Entity, &Window, &RawHandleWrapper, Option<&PrimaryWindow>)>>,
    mut removed: Extract<RemovedComponents<RawHandleWrapper>>,
    mut window_surfaces: ResMut<WindowSurfaces>,
) {
    for (entity, window, handle, primary) in windows.iter() {
        if primary.is_some() {
            extracted_windows.primary = Some(entity);
        }

        let (new_width, new_height) = (
            window.resolution.physical_width().max(1),
            window.resolution.physical_height().max(1),
        );

        let extracted_window = extracted_windows.entry(entity).or_insert(ExtractedWindow {
            entity,
            handle: handle.clone(),
            physical_width: new_width,
            physical_height: new_height,
            present_mode: window.present_mode,
            desired_maximum_frame_latency: window.desired_maximum_frame_latency,
            swap_chain_texture: None,
            swap_chain_texture_view: None,
            size_changed: false,
            swap_chain_texture_format: None,
            swap_chain_texture_view_format: None,
            present_mode_changed: false,
            alpha_mode: window.composite_alpha_mode,
            needs_initial_present: true,
        });

        if extracted_window.swap_chain_texture.is_none() {
            // If we called present on the previous swap-chain texture last update,
            // then drop the swap chain frame here, otherwise we can keep it for the
            // next update as an optimization. `prepare_windows` will only acquire a new
            // swap chain texture if needed.
            extracted_window.swap_chain_texture_view = None;
        }
        extracted_window.size_changed = new_width != extracted_window.physical_width
            || new_height != extracted_window.physical_height;
        extracted_window.present_mode_changed =
            window.present_mode != extracted_window.present_mode;

        if extracted_window.size_changed {
            debug!(
                "Window size changed from {}x{} to {}x{}",
                extracted_window.physical_width,
                extracted_window.physical_height,
                new_width,
                new_height
            );
            extracted_window.physical_width = new_width;
            extracted_window.physical_height = new_height;
        }

        if extracted_window.present_mode_changed {
            debug!(
                "Window Present Mode changed from {:?} to {:?}",
                extracted_window.present_mode, window.present_mode
            );
            extracted_window.present_mode = window.present_mode;
        }
    }

    for closing_window in closing.read() {
        extracted_windows.remove(&closing_window.window);
        window_surfaces.remove(&closing_window.window);
    }
    for removed_window in removed.read() {
        extracted_windows.remove(&removed_window);
        window_surfaces.remove(&removed_window);
    }
}

struct SurfaceData {
    // TODO: what lifetime should this be?
    surface: WgpuWrapper<wgpu::Surface<'static>>,
    configuration: SurfaceConfiguration,
    texture_view_format: Option<TextureFormat>,
}

#[derive(Resource, Default)]
pub struct WindowSurfaces {
    surfaces: EntityHashMap<SurfaceData>,
    /// List of windows that we have already called the initial `configure_surface` for
    configured_windows: HashSet<Entity>,
}

impl WindowSurfaces {
    fn remove(&mut self, window: &Entity) {
        self.surfaces.remove(window);
        self.configured_windows.remove(window);
    }
}

/// (re)configures window surfaces, and obtains a swapchain texture for rendering.
///
/// NOTE: `get_current_texture` in `prepare_windows` can take a long time if the GPU workload is
/// the performance bottleneck. This can be seen in profiles as multiple prepare-set systems all
/// taking an unusually long time to complete, and all finishing at about the same time as the
/// `prepare_windows` system. Improvements in bevy are planned to avoid this happening when it
/// should not but it will still happen as it is easy for a user to create a large GPU workload
/// relative to the GPU performance and/or CPU workload.
/// This can be caused by many reasons, but several of them are:
/// - GPU workload is more than your current GPU can manage
/// - Error / performance bug in your custom shaders
/// - wgpu was unable to detect a proper GPU hardware-accelerated device given the chosen
///   [`Backends`](crate::settings::Backends), [`WgpuLimits`](crate::settings::WgpuLimits),
///   and/or [`WgpuFeatures`](crate::settings::WgpuFeatures). For example, on Windows currently
///   `DirectX 11` is not supported by wgpu 0.12 and so if your GPU/drivers do not support Vulkan,
///   it may be that a software renderer called "Microsoft Basic Render Driver" using `DirectX 12`
///   will be chosen and performance will be very poor. This is visible in a log message that is
///   output during renderer initialization.
///   Another alternative is to try to use [`ANGLE`](https://github.com/gfx-rs/wgpu#angle) and
///   [`Backends::GL`](crate::settings::Backends::GL) with the `gles` feature enabled if your
///   GPU/drivers support `OpenGL 4.3` / `OpenGL ES 3.0` or later.
pub fn prepare_windows(
    mut windows: ResMut<ExtractedWindows>,
    mut window_surfaces: ResMut<WindowSurfaces>,
    render_device: Res<RenderDevice>,
    cameras: Query<&ExtractedCamera>,
    offscreen_blit: Option<Res<OffscreenBlitState>>,
    #[cfg(target_os = "linux")] render_instance: Res<RenderInstance>,
) {
    let has_offscreen_blit = offscreen_blit.is_some();

    // If all windows are gone, signal the offscreen blit to stop and detach
    // the Metal layer so any in-progress nextDrawable returns nil immediately.
    #[cfg(target_os = "macos")]
    if windows.windows.is_empty() {
        if let Some(ref state) = offscreen_blit {
            if !state.shutting_down.swap(true, core::sync::atomic::Ordering::SeqCst) {
                if let Some(ref ms) = state.metal_state {
                    #[allow(unused_imports)]
                    use objc::{msg_send, sel, sel_impl};
                    unsafe {
                        let _: () = msg_send![ms.layer_ptr, removeFromSuperlayer];
                    }
                }
            }
        }
    }

    for window in windows.windows.values_mut() {
        let window_surfaces = window_surfaces.deref_mut();
        let Some(surface_data) = window_surfaces.surfaces.get(&window.entity) else {
            continue;
        };

        // Skip drawable acquisition if no camera targets this window.
        // When offscreen blit is active, also skip needs_initial_present —
        // the raw Metal presenter is the sole layer owner.
        let any_camera_targets_window = cameras.iter().any(|cam| {
            matches!(
                &cam.target,
                Some(NormalizedRenderTarget::Window(w)) if w.entity() == window.entity
            )
        });
        if !any_camera_targets_window && (!window.needs_initial_present || has_offscreen_blit) {
            window.swap_chain_texture_format = Some(surface_data.configuration.format);
            if has_offscreen_blit {
                window.needs_initial_present = false;
            }
            continue;
        }

        // We didn't present the previous frame, so we can keep using our existing swapchain texture.
        if window.has_swapchain_texture() && !window.size_changed && !window.present_mode_changed {
            continue;
        }

        // A recurring issue is hitting `wgpu::SurfaceError::Timeout` on certain Linux
        // mesa driver implementations. This seems to be a quirk of some drivers.
        // We'd rather keep panicking when not on Linux mesa, because in those case,
        // the `Timeout` is still probably the symptom of a degraded unrecoverable
        // application state.
        // see https://github.com/bevyengine/bevy/pull/5957
        // and https://github.com/gfx-rs/wgpu/issues/1218
        #[cfg(target_os = "linux")]
        let may_erroneously_timeout = || {
            render_instance
                .enumerate_adapters(wgpu::Backends::VULKAN)
                .iter()
                .any(|adapter| {
                    let name = adapter.get_info().name;
                    name.starts_with("Radeon")
                        || name.starts_with("AMD")
                        || name.starts_with("Intel")
                })
        };

        let surface = &surface_data.surface;
        match surface.get_current_texture() {
            Ok(frame) => {
                window.set_swapchain_texture(frame);
            }
            Err(wgpu::SurfaceError::Outdated) => {
                render_device.configure_surface(surface, &surface_data.configuration);
                let frame = match surface.get_current_texture() {
                    Ok(frame) => frame,
                    Err(err) => {
                        // This is a common occurrence on X11 and Xwayland with NVIDIA drivers
                        // when opening and resizing the window.
                        warn!("Couldn't get swap chain texture after configuring. Cause: '{err}'");
                        continue;
                    }
                };
                window.set_swapchain_texture(frame);
            }
            #[cfg(target_os = "linux")]
            Err(wgpu::SurfaceError::Timeout) if may_erroneously_timeout() => {
                tracing::trace!(
                    "Couldn't get swap chain texture. This is probably a quirk \
                        of your Linux GPU driver, so it can be safely ignored."
                );
            }
            Err(err) => {
                panic!("Couldn't get swap chain texture, operation unrecoverable: {err}");
            }
        }
        window.swap_chain_texture_format = Some(surface_data.configuration.format);
    }
}

pub fn need_surface_configuration(
    windows: Res<ExtractedWindows>,
    window_surfaces: Res<WindowSurfaces>,
) -> bool {
    for window in windows.windows.values() {
        if !window_surfaces.configured_windows.contains(&window.entity)
            || window.size_changed
            || window.present_mode_changed
        {
            return true;
        }
    }
    false
}

// 2 is wgpu's default/what we've been using so far.
// 1 is the minimum, but may cause lower framerates due to the cpu waiting for the gpu to finish
// all work for the previous frame before starting work on the next frame, which then means the gpu
// has to wait for the cpu to finish to start on the next frame.
const DEFAULT_DESIRED_MAXIMUM_FRAME_LATENCY: u32 = 2;

/// Creates window surfaces.
pub fn create_surfaces(
    // By accessing a NonSend resource, we tell the scheduler to put this system on the main thread,
    // which is necessary for some OS's
    #[cfg(any(target_os = "macos", target_os = "ios"))] _marker: bevy_ecs::system::NonSendMarker,
    mut windows: ResMut<ExtractedWindows>,
    mut window_surfaces: ResMut<WindowSurfaces>,
    render_instance: Res<RenderInstance>,
    render_adapter: Res<RenderAdapter>,
    render_device: Res<RenderDevice>,
) {
    for window in windows.windows.values_mut() {
        let data = window_surfaces
            .surfaces
            .entry(window.entity)
            .or_insert_with(|| {
                let surface_target = SurfaceTargetUnsafe::RawHandle {
                    raw_display_handle: window.handle.get_display_handle(),
                    raw_window_handle: window.handle.get_window_handle(),
                };
                // SAFETY: The window handles in ExtractedWindows will always be valid objects to create surfaces on
                let surface = unsafe {
                    // NOTE: On some OSes this MUST be called from the main thread.
                    // As of wgpu 0.15, only fallible if the given window is a HTML canvas and obtaining a WebGPU or WebGL2 context fails.
                    render_instance
                        .create_surface_unsafe(surface_target)
                        .expect("Failed to create wgpu surface")
                };
                let caps = surface.get_capabilities(&render_adapter);
                let present_mode = present_mode(window, &caps);
                let formats = caps.formats;
                // For future HDR output support, we'll need to request a format that supports HDR,
                // but as of wgpu 0.15 that is not yet supported.
                // Prefer sRGB formats for surfaces, but fall back to first available format if no sRGB formats are available.
                let mut format = *formats.first().expect("No supported formats for surface");
                for available_format in formats {
                    // Rgba8UnormSrgb and Bgra8UnormSrgb and the only sRGB formats wgpu exposes that we can use for surfaces.
                    if available_format == TextureFormat::Rgba8UnormSrgb
                        || available_format == TextureFormat::Bgra8UnormSrgb
                    {
                        format = available_format;
                        break;
                    }
                }

                let texture_view_format = if !format.is_srgb() {
                    Some(format.add_srgb_suffix())
                } else {
                    None
                };
                let configuration = SurfaceConfiguration {
                    format,
                    width: window.physical_width,
                    height: window.physical_height,
                    usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_DST,
                    present_mode,
                    desired_maximum_frame_latency: window
                        .desired_maximum_frame_latency
                        .map(NonZero::<u32>::get)
                        .unwrap_or(DEFAULT_DESIRED_MAXIMUM_FRAME_LATENCY),
                    alpha_mode: match window.alpha_mode {
                        CompositeAlphaMode::Auto => wgpu::CompositeAlphaMode::Auto,
                        CompositeAlphaMode::Opaque => wgpu::CompositeAlphaMode::Opaque,
                        CompositeAlphaMode::PreMultiplied => {
                            wgpu::CompositeAlphaMode::PreMultiplied
                        }
                        CompositeAlphaMode::PostMultiplied => {
                            wgpu::CompositeAlphaMode::PostMultiplied
                        }
                        CompositeAlphaMode::Inherit => wgpu::CompositeAlphaMode::Inherit,
                    },
                    view_formats: match texture_view_format {
                        Some(format) => vec![format],
                        None => vec![],
                    },
                };

                render_device.configure_surface(&surface, &configuration);

                SurfaceData {
                    surface: WgpuWrapper::new(surface),
                    configuration,
                    texture_view_format,
                }
            });

        if window.size_changed || window.present_mode_changed {
            // normally this is dropped on present but we double check here to be safe as failure to
            // drop it will cause validation errors in wgpu
            drop(window.swap_chain_texture.take());
            #[cfg_attr(
                target_arch = "wasm32",
                expect(clippy::drop_non_drop, reason = "texture views are not drop on wasm")
            )]
            drop(window.swap_chain_texture_view.take());

            data.configuration.width = window.physical_width;
            data.configuration.height = window.physical_height;
            let caps = data.surface.get_capabilities(&render_adapter);
            data.configuration.present_mode = present_mode(window, &caps);
            render_device.configure_surface(&data.surface, &data.configuration);
        }

        window_surfaces.configured_windows.insert(window.entity);
    }
}

fn present_mode(
    window: &mut ExtractedWindow,
    caps: &wgpu::SurfaceCapabilities,
) -> wgpu::PresentMode {
    let present_mode = match window.present_mode {
        PresentMode::Fifo => wgpu::PresentMode::Fifo,
        PresentMode::FifoRelaxed => wgpu::PresentMode::FifoRelaxed,
        PresentMode::Mailbox => wgpu::PresentMode::Mailbox,
        PresentMode::Immediate => wgpu::PresentMode::Immediate,
        PresentMode::AutoVsync => wgpu::PresentMode::AutoVsync,
        PresentMode::AutoNoVsync => wgpu::PresentMode::AutoNoVsync,
    };
    let fallbacks = match present_mode {
        wgpu::PresentMode::AutoVsync => {
            &[wgpu::PresentMode::FifoRelaxed, wgpu::PresentMode::Fifo][..]
        }
        wgpu::PresentMode::AutoNoVsync => &[
            wgpu::PresentMode::Immediate,
            wgpu::PresentMode::Mailbox,
            wgpu::PresentMode::Fifo,
        ][..],
        wgpu::PresentMode::Mailbox => &[
            wgpu::PresentMode::Mailbox,
            wgpu::PresentMode::Immediate,
            wgpu::PresentMode::Fifo,
        ][..],
        // Always end in FIFO to make sure it's always supported
        x => &[x, wgpu::PresentMode::Fifo][..],
    };
    let new_present_mode = fallbacks
        .iter()
        .copied()
        .find(|fallback| caps.present_modes.contains(fallback))
        .unwrap_or_else(|| {
            unreachable!(
                "Fallback system failed to choose present mode. \
                            This is a bug. Mode: {:?}, Options: {:?}",
                window.present_mode, &caps.present_modes
            );
        });
    if new_present_mode != present_mode && fallbacks.contains(&present_mode) {
        info!("PresentMode {present_mode:?} requested but not available. Falling back to {new_present_mode:?}");
    }
    new_present_mode
}

/// Insert this resource into the MAIN world to enable offscreen rendering.
/// The WindowRenderPlugin will create the offscreen texture in `finish()` and
/// blit it to the window surface each frame, bypassing the normal drawable
/// acquisition path.
#[derive(Resource, Clone)]
pub struct OffscreenPresentConfig {
    /// The ManualTextureViewHandle that cameras should target.
    pub handle: bevy_camera::ManualTextureViewHandle,
    /// Physical width of the offscreen render target in pixels.
    pub width: u32,
    /// Physical height of the offscreen render target in pixels.
    pub height: u32,
    /// Scale factor (physical / logical). Set to 2.0 for Retina displays.
    pub scale_factor: f32,
}

#[derive(Resource)]
pub(crate) struct OffscreenBlitState {
    source_texture: wgpu::Texture,
    #[cfg(target_os = "macos")]
    metal_state: Option<MetalPresentState>,
    pub(crate) shutting_down: alloc::sync::Arc<core::sync::atomic::AtomicBool>,
}

#[cfg(target_os = "macos")]
struct MetalPresentState {
    /// Shared with wgpu — ensures GPU ordering between render and blit.
    command_queue: alloc::sync::Arc<parking_lot::Mutex<metal::CommandQueue>>,
    source_metal_texture: metal::Texture,
    layer_ptr: *mut objc::runtime::Object,
}

#[cfg(target_os = "macos")]
unsafe impl Send for MetalPresentState {}
#[cfg(target_os = "macos")]
unsafe impl Sync for MetalPresentState {}

/// Non-macOS fallback: blit via wgpu surface.
#[cfg(not(target_os = "macos"))]
fn blit_offscreen_to_window(
    state: Option<Res<OffscreenBlitState>>,
    mut windows: ResMut<ExtractedWindows>,
    mut window_surfaces: ResMut<WindowSurfaces>,
    render_device: Res<RenderDevice>,
    render_queue: Res<crate::renderer::RenderQueue>,
) {
    let Some(state) = state else { return };

    for window in windows.windows.values_mut() {
        let window_surfaces = window_surfaces.deref_mut();
        let Some(surface_data) = window_surfaces.surfaces.get(&window.entity) else {
            continue;
        };
        if window.swap_chain_texture.is_some() { continue; }

        let frame = match surface_data.surface.get_current_texture() {
            Ok(frame) => frame,
            Err(wgpu::SurfaceError::Outdated) => {
                render_device.configure_surface(&surface_data.surface, &surface_data.configuration);
                match surface_data.surface.get_current_texture() {
                    Ok(f) => f,
                    Err(_) => continue,
                }
            }
            Err(_) => continue,
        };

        let src_size = state.source_texture.size();
        let mut encoder = render_device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("offscreen_blit") },
        );
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &state.source_texture, mip_level: 0,
                origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &frame.texture, mip_level: 0,
                origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: src_size.width.min(frame.texture.width()),
                height: src_size.height.min(frame.texture.height()),
                depth_or_array_layers: 1,
            },
        );
        render_queue.submit(core::iter::once(encoder.finish()));
        frame.present();
    }
}

/// One-shot system: grabs the CAMetalLayer pointer on the main thread.
/// NonSendMarker ensures this runs on the main thread (required for NSView access).
/// Only runs once — inserts MetalPresentState into OffscreenBlitState, then the
/// blit system (which runs on the render thread) takes over.
#[cfg(target_os = "macos")]
fn init_metal_presenter(
    _marker: bevy_ecs::system::NonSendMarker,
    mut state: Option<ResMut<OffscreenBlitState>>,
    windows: Res<ExtractedWindows>,
    render_queue: Res<crate::renderer::RenderQueue>,
) {
    #[allow(unused_imports)]
    use objc::{msg_send, sel, sel_impl, class};

    let Some(ref mut state) = state else { return };
    if state.metal_state.is_some() { return; }

    let Some(window) = windows.windows.values().next() else { return };

    let raw_handle = window.handle.get_window_handle();
    let ns_view = match raw_handle {
        wgpu::rwh::RawWindowHandle::AppKit(handle) => handle.ns_view.as_ptr(),
        _ => return,
    };

    let ns_view_ptr = ns_view as *mut objc::runtime::Object;

    let root_layer: *mut objc::runtime::Object = unsafe {
        objc::msg_send![ns_view_ptr, layer]
    };
    if root_layer.is_null() { return; }

    let is_metal: bool = unsafe {
        objc::msg_send![root_layer, isKindOfClass: objc::class!(CAMetalLayer)]
    };
    let layer_ptr = if is_metal {
        root_layer
    } else {
        let sublayers: *mut objc::runtime::Object = unsafe {
            objc::msg_send![root_layer, sublayers]
        };
        if sublayers.is_null() { return; }
        let count: usize = unsafe { objc::msg_send![sublayers, count] };
        let mut found: *mut objc::runtime::Object = core::ptr::null_mut();
        for i in 0..count {
            let sub: *mut objc::runtime::Object = unsafe {
                objc::msg_send![sublayers, objectAtIndex: i]
            };
            let is_metal_sub: bool = unsafe {
                objc::msg_send![sub, isKindOfClass: objc::class!(CAMetalLayer)]
            };
            if is_metal_sub {
                found = sub;
                break;
            }
        }
        if found.is_null() { return; }
        found
    };

    // Triple buffering
    unsafe { let () = objc::msg_send![layer_ptr, setMaximumDrawableCount: 3u64]; }

    // Get wgpu's Metal command queue — ensures GPU ordering between render and blit
    let command_queue = unsafe {
        render_queue.as_hal::<wgpu::hal::api::Metal>()
            .map(|hal_queue| hal_queue.as_raw().clone())
    }.expect("Failed to get Metal command queue from wgpu");

    let source_metal_texture = unsafe {
        state.source_texture.as_hal::<wgpu::hal::api::Metal>()
            .map(|hal_tex| hal_tex.raw_handle().to_owned())
    }.expect("Failed to get Metal texture from wgpu texture");

    info!(
        "Metal presenter initialized: shared queue, src={}x{}, triple-buffered",
        source_metal_texture.width(), source_metal_texture.height()
    );

    state.metal_state = Some(MetalPresentState {
        command_queue,
        source_metal_texture,
        layer_ptr,
    });
}

/// Render-thread blit: acquires drawable, blits offscreen texture, presents.
/// No NonSendMarker — runs on the render thread. Metal nextDrawable/present
/// are thread-safe; only the initial NSView layer access needed the main thread.
#[cfg(target_os = "macos")]
fn blit_offscreen_to_window(
    state: Option<Res<OffscreenBlitState>>,
) {
    #[allow(unused_imports)]
    use objc::{msg_send, sel, sel_impl, class};
    use objc::rc::autoreleasepool;

    let Some(state) = state else { return };
    if state.shutting_down.load(core::sync::atomic::Ordering::Relaxed) { return; }
    let Some(ref metal_state) = state.metal_state else { return };

    let t0 = bevy_platform::time::Instant::now();

    autoreleasepool(|| {
        if state.shutting_down.load(core::sync::atomic::Ordering::Relaxed) { return; }
        // Use a short timeout so the render thread can check the shutdown flag
        // and exit cleanly rather than blocking indefinitely on a destroyed layer.
        let timeout_s: f64 = 0.1;
        let drawable: *mut objc::runtime::Object = unsafe {
            objc::msg_send![metal_state.layer_ptr, nextDrawableWithTimeout: timeout_s]
        };
        if drawable.is_null() { return; }

        let t_drawable = bevy_platform::time::Instant::now();

        let dst_texture: *mut objc::runtime::Object = unsafe {
            objc::msg_send![drawable, texture]
        };
        let dst = unsafe { &*(dst_texture as *const metal::TextureRef) };

        let queue_lock = metal_state.command_queue.lock();
        let cmd_buf = queue_lock.new_command_buffer();
        let blit_encoder = cmd_buf.new_blit_command_encoder();

        let src = &metal_state.source_metal_texture;
        let w = src.width().min(dst.width());
        let h = src.height().min(dst.height());

        blit_encoder.copy_from_texture(
            src, 0, 0,
            metal::MTLOrigin { x: 0, y: 0, z: 0 },
            metal::MTLSize { width: w, height: h, depth: 1 },
            dst, 0, 0,
            metal::MTLOrigin { x: 0, y: 0, z: 0 },
        );
        blit_encoder.end_encoding();

        let drawable_ref = unsafe { &*(drawable as *const metal::DrawableRef) };
        cmd_buf.present_drawable(drawable_ref);
        cmd_buf.commit();

        let t_done = bevy_platform::time::Instant::now();

        let drawable_ms = (t_drawable - t0).as_secs_f64() * 1000.0;
        let blit_ms = (t_done - t_drawable).as_secs_f64() * 1000.0;
        let total_ms = (t_done - t0).as_secs_f64() * 1000.0;

        if total_ms > 2.0 {
            trace!(
                "METAL_BLIT: {:.1}ms total | drawable={:.1} blit+present={:.1}",
                total_ms, drawable_ms, blit_ms
            );
        }
    });
}
