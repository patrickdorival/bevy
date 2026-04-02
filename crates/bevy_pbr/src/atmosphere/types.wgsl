#define_import_path bevy_pbr::atmosphere::types

struct Atmosphere {
    ground_albedo: vec3<f32>,
    // Radius of the planet surface (for scattering ground)
    bottom_radius: f32, // units: m
    // Radius at which we consider the atmosphere to 'end' for our calculations (from center of planet)
    top_radius: f32, // units: m
    // Direction from planet centre toward the camera, in world space.
    // For flat Y-up worlds: vec3(0, 1, 0). For spherical planets: normalised
    // direction from planet centre to camera, updated each frame.
    planet_up: vec3<f32>,
    // Distance from camera/observer to planet centre. Used for camera position
    // reconstruction (planet_up * observer_radius). Separate from bottom_radius
    // so scattering uses planet surface while positioning uses observer altitude.
    observer_radius: f32,
}

struct AtmosphereSettings {
    transmittance_lut_size: vec2<u32>,
    multiscattering_lut_size: vec2<u32>,
    sky_view_lut_size: vec2<u32>,
    aerial_view_lut_size: vec3<u32>,
    transmittance_lut_samples: u32,
    multiscattering_lut_dirs: u32,
    multiscattering_lut_samples: u32,
    sky_view_lut_samples: u32,
    aerial_view_lut_samples: u32,
    aerial_view_lut_max_distance: f32,
    scene_units_to_m: f32,
    sky_max_samples: u32,
    rendering_method: u32,
}

// "Atmosphere space" is the camera's local coordinate system where Y is always
// "up" from the planet surface. The world_from_atmosphere matrix maps this
// local space to world space. For spherical planets, set Atmosphere.planet_up
// on the Rust side to the direction from planet centre to camera — the
// prepare_atmosphere_transforms system builds the correct basis from it.
struct AtmosphereTransforms {
    world_from_atmosphere: mat4x4<f32>,
}

struct AtmosphereData {
    atmosphere: Atmosphere,
    settings: AtmosphereSettings,
}