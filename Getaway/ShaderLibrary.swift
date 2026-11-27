import Metal

/// Loads the compiled shader library. The Swift Playgrounds variant replaces this file
/// with one that compiles the same shaders from source at launch (see scripts/make-playground.sh).
enum ShaderLibrary {
    static func make(device: MTLDevice) throws -> MTLLibrary {
        guard let library = device.makeDefaultLibrary() else {
            throw RendererError.resourceCreation("shader library")
        }
        return library
    }
}
