import Foundation
import simd

struct FlyCamera {
    static let startPosition = SIMD3<Float>(800, 90, 800)

    var position = FlyCamera.startPosition
    /// Radians; 0 looks down -Z, positive turns right.
    var yaw: Float = 2.4
    var pitch: Float = -0.18
    var speed: Float = 90
    var lookSensitivity: Float = 0.0025

    var forward: SIMD3<Float> {
        SIMD3(cos(pitch) * sin(yaw), sin(pitch), -cos(pitch) * cos(yaw))
    }

    var right: SIMD3<Float> {
        SIMD3(cos(yaw), 0, sin(yaw))
    }

    var viewMatrix: simd_float4x4 {
        .lookAt(eye: position, forward: forward, up: SIMD3(0, 1, 0))
    }

    mutating func reset() {
        self = FlyCamera(speed: speed, lookSensitivity: lookSensitivity)
    }

    mutating func update(deltaTime dt: Float, input: InputState) {
        let look = input.consumeLook()
        yaw += look.x * lookSensitivity
        pitch = min(max(pitch - look.y * lookSensitivity, -1.55), 1.55)

        let scroll = input.consumeScroll()
        if scroll != 0 {
            speed = min(max(speed * pow(1.1, scroll), 2), 4000)
        }

        var move = SIMD3<Float>.zero
        if input.isHeld(.forward)  { move += forward }
        if input.isHeld(.backward) { move -= forward }
        if input.isHeld(.right)    { move += right }
        if input.isHeld(.left)     { move -= right }
        if input.isHeld(.up)       { move.y += 1 }
        if input.isHeld(.down)     { move.y -= 1 }

        guard length_squared(move) > 0 else { return }
        var currentSpeed = speed
        if input.isHeld(.boost) { currentSpeed *= 5 }
        if input.isHeld(.slow)  { currentSpeed *= 0.2 }
        position += normalize(move) * currentSpeed * dt
    }
}
