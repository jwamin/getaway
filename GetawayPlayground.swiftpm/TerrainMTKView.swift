import MetalKit

#if os(macOS)
import AppKit

final class TerrainMTKView: MTKView {
    let input = InputState()

    override var acceptsFirstResponder: Bool { true }

    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        NotificationCenter.default.removeObserver(self)
        guard let window else { return }
        window.acceptsMouseMovedEvents = true
        window.makeFirstResponder(self)
        NotificationCenter.default.addObserver(self, selector: #selector(windowLostFocus),
                                               name: NSWindow.didResignKeyNotification, object: window)
    }

    @objc private func windowLostFocus() {
        setMouseCaptured(false)
        input.releaseAll()
    }

    private func setMouseCaptured(_ captured: Bool) {
        guard captured != input.mouseCaptured else { return }
        input.mouseCaptured = captured
        if captured {
            NSCursor.hide()
            CGAssociateMouseAndMouseCursorPosition(0)
        } else {
            CGAssociateMouseAndMouseCursorPosition(1)
            NSCursor.unhide()
        }
    }

    // MARK: Keyboard (key codes are physical positions, so WASD works on any layout)

    private static let heldKeys: [UInt16: InputKey] = [
        13: .forward, 126: .forward,   // W, Up arrow
        1: .backward, 125: .backward,  // S, Down arrow
        0: .left, 123: .left,          // A, Left arrow
        2: .right, 124: .right,        // D, Right arrow
        14: .up, 49: .up,              // E, Space
        12: .down, 8: .down,           // Q, C
    ]

    private static let actions: [UInt16: InputAction] = [
        3: .toggleWireframe,   // F
        4: .toggleHelp,        // H
        37: .toggleLODFreeze,  // L
        24: .increaseDetail,   // =
        27: .decreaseDetail,   // -
        30: .faster,           // ]
        33: .slower,           // [
        15: .resetCamera,      // R
    ]

    override func keyDown(with event: NSEvent) {
        if event.keyCode == 53 { // Escape
            setMouseCaptured(false)
            return
        }
        if let key = Self.heldKeys[event.keyCode] {
            input.press(key)
        } else if !event.isARepeat, let action = Self.actions[event.keyCode] {
            input.trigger(action)
        }
    }

    override func keyUp(with event: NSEvent) {
        if let key = Self.heldKeys[event.keyCode] {
            input.release(key)
        }
    }

    override func flagsChanged(with event: NSEvent) {
        let flags = event.modifierFlags
        flags.contains(.shift) ? input.press(.boost) : input.release(.boost)
        flags.contains(.option) ? input.press(.slow) : input.release(.slow)
    }

    // MARK: Mouse

    override func mouseDown(with event: NSEvent) {
        window?.makeFirstResponder(self)
        setMouseCaptured(true)
    }

    override func mouseMoved(with event: NSEvent) {
        if input.mouseCaptured { addLook(event) }
    }

    override func mouseDragged(with event: NSEvent) { addLook(event) }
    override func rightMouseDragged(with event: NSEvent) { addLook(event) }
    override func otherMouseDragged(with event: NSEvent) { addLook(event) }

    private func addLook(_ event: NSEvent) {
        input.addLook(dx: Float(event.deltaX), dy: Float(event.deltaY))
    }

    override func scrollWheel(with event: NSEvent) {
        let scale: Float = event.hasPreciseScrollingDeltas ? 0.1 : 1
        input.addScroll(Float(event.scrollingDeltaY) * scale)
    }
}

#else
import UIKit

final class TerrainMTKView: MTKView {
    let input = InputState()
    private var activeTouches = 0

    override init(frame frameRect: CGRect, device: MTLDevice?) {
        super.init(frame: frameRect, device: device)
        isMultipleTouchEnabled = true

        let scroll = UIPanGestureRecognizer(target: self, action: #selector(handleScroll(_:)))
        scroll.allowedScrollTypesMask = .all
        scroll.allowedTouchTypes = []
        addGestureRecognizer(scroll)
    }

    required init(coder: NSCoder) {
        super.init(coder: coder)
    }

    override var canBecomeFirstResponder: Bool { true }

    override func didMoveToWindow() {
        super.didMoveToWindow()
        if window != nil { becomeFirstResponder() }
    }

    // MARK: Hardware keyboard

    private static let heldKeys: [UIKeyboardHIDUsage: InputKey] = [
        .keyboardW: .forward, .keyboardUpArrow: .forward,
        .keyboardS: .backward, .keyboardDownArrow: .backward,
        .keyboardA: .left, .keyboardLeftArrow: .left,
        .keyboardD: .right, .keyboardRightArrow: .right,
        .keyboardE: .up, .keyboardSpacebar: .up,
        .keyboardQ: .down, .keyboardC: .down,
        .keyboardLeftShift: .boost, .keyboardRightShift: .boost,
        .keyboardLeftAlt: .slow, .keyboardRightAlt: .slow,
    ]

    private static let actions: [UIKeyboardHIDUsage: InputAction] = [
        .keyboardF: .toggleWireframe,
        .keyboardH: .toggleHelp,
        .keyboardL: .toggleLODFreeze,
        .keyboardEqualSign: .increaseDetail,
        .keyboardHyphen: .decreaseDetail,
        .keyboardCloseBracket: .faster,
        .keyboardOpenBracket: .slower,
        .keyboardR: .resetCamera,
    ]

    override func pressesBegan(_ presses: Set<UIPress>, with event: UIPressesEvent?) {
        var unhandled = Set<UIPress>()
        for press in presses {
            guard let code = press.key?.keyCode else { unhandled.insert(press); continue }
            if let key = Self.heldKeys[code] {
                input.press(key)
            } else if let action = Self.actions[code] {
                input.trigger(action)
            } else {
                unhandled.insert(press)
            }
        }
        if !unhandled.isEmpty { super.pressesBegan(unhandled, with: event) }
    }

    override func pressesEnded(_ presses: Set<UIPress>, with event: UIPressesEvent?) {
        releaseKeys(for: presses)
        super.pressesEnded(presses, with: event)
    }

    override func pressesCancelled(_ presses: Set<UIPress>, with event: UIPressesEvent?) {
        releaseKeys(for: presses)
        super.pressesCancelled(presses, with: event)
    }

    private func releaseKeys(for presses: Set<UIPress>) {
        for press in presses {
            if let code = press.key?.keyCode, let key = Self.heldKeys[code] {
                input.release(key)
            }
        }
    }

    // MARK: Touch / pointer: drag to look, hold two fingers to fly forward

    override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?) {
        becomeFirstResponder()
        activeTouches += touches.count
        if activeTouches >= 2 { input.press(.forward) }
    }

    override func touchesMoved(_ touches: Set<UITouch>, with event: UIEvent?) {
        guard let touch = touches.first else { return }
        let location = touch.location(in: self)
        let previous = touch.previousLocation(in: self)
        input.addLook(dx: Float(location.x - previous.x), dy: Float(location.y - previous.y))
    }

    override func touchesEnded(_ touches: Set<UITouch>, with event: UIEvent?) {
        endTouches(touches)
    }

    override func touchesCancelled(_ touches: Set<UITouch>, with event: UIEvent?) {
        endTouches(touches)
    }

    private func endTouches(_ touches: Set<UITouch>) {
        activeTouches = max(activeTouches - touches.count, 0)
        if activeTouches < 2 { input.release(.forward) }
    }

    @objc private func handleScroll(_ gesture: UIPanGestureRecognizer) {
        let translation = gesture.translation(in: self)
        input.addScroll(Float(-translation.y) * 0.05)
        gesture.setTranslation(.zero, in: self)
    }
}
#endif
