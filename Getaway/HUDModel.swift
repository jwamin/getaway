import Observation

@Observable
final class HUDModel {
    var fps: Double = 0
    var speed: Float = 0
    var altitude: Float = 0
    var lodScale: Float = 0
    var maxTessellation: Int = 0
    var wireframe = false
    var lodFrozen = false
    var showHelp = true
    var mouseCaptured = false
    var errorMessage: String?
}
