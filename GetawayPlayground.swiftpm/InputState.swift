import simd

enum InputKey: Hashable {
    case forward, backward, left, right, up, down, boost, slow
}

enum InputAction {
    case toggleWireframe
    case toggleHelp
    case toggleLODFreeze
    case increaseDetail
    case decreaseDetail
    case faster
    case slower
    case resetCamera
}

/// Platform-neutral input collected by the view and drained by the renderer each frame.
final class InputState {
    private(set) var heldKeys: Set<InputKey> = []
    private var lookDelta = SIMD2<Float>.zero
    private var scrollDelta: Float = 0
    private var pendingActions: [InputAction] = []
    var mouseCaptured = false

    func press(_ key: InputKey) { heldKeys.insert(key) }
    func release(_ key: InputKey) { heldKeys.remove(key) }
    func releaseAll() { heldKeys.removeAll() }
    func isHeld(_ key: InputKey) -> Bool { heldKeys.contains(key) }

    func addLook(dx: Float, dy: Float) { lookDelta += SIMD2(dx, dy) }
    func addScroll(_ amount: Float) { scrollDelta += amount }
    func trigger(_ action: InputAction) { pendingActions.append(action) }

    func consumeLook() -> SIMD2<Float> {
        defer { lookDelta = .zero }
        return lookDelta
    }

    func consumeScroll() -> Float {
        defer { scrollDelta = 0 }
        return scrollDelta
    }

    func consumeActions() -> [InputAction] {
        defer { pendingActions.removeAll() }
        return pendingActions
    }
}
