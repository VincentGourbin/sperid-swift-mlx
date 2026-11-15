// MLXThread.swift - one long-lived thread for every MLX call of an engine
// Copyright 2026 Vincent Gourbin

import Foundation

/// mlx-swift must build and evaluate a graph on the same thread (pattern MLX-004): a serial DispatchQueue
/// keeps the order of its blocks but not their thread, which crashed Idol Studio in the Metal command
/// encoder. Every `IdolEngine` owns one of these and runs all its MLX work on it.
public final class MLXThread: @unchecked Sendable {
    private let condition = NSCondition()
    private var jobs: [() -> Void] = []
    private var thread: Thread!

    public init(name: String = "idol.mlx") {
        thread = Thread { [unowned self] in self.loop() }
        thread.name = name
        thread.qualityOfService = .userInitiated
        thread.stackSize = 16 << 20
        thread.start()
    }

    public func async(_ job: @escaping () -> Void) {
        condition.lock(); jobs.append(job); condition.signal(); condition.unlock()
    }

    /// Runs `body` on the MLX thread and returns its result to the caller's task.
    public func run<T: Sendable>(_ body: @escaping () throws -> T) async throws -> T {
        try await withCheckedThrowingContinuation { continuation in
            async { continuation.resume(with: Result { try body() }) }
        }
    }

    private func loop() {
        while true {
            condition.lock()
            while jobs.isEmpty { condition.wait() }
            let job = jobs.removeFirst()
            condition.unlock()
            autoreleasepool { job() }
        }
    }
}

/// A cancellation flag shared between a Swift task and the MLX thread.
final class CancellationFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var value = false
    func cancel() { lock.lock(); value = true; lock.unlock() }
    var isCancelled: Bool { lock.lock(); defer { lock.unlock() }; return value }
}
