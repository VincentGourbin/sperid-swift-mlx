// MLXThread.swift - one long-lived thread for every MLX call of an engine
// Copyright 2026 Vincent Gourbin

import Foundation

/// mlx-swift must build and evaluate a graph on the same thread (pattern MLX-004): a serial DispatchQueue
/// keeps the order of its blocks but not their thread, which crashed Idol Studio in the Metal command
/// encoder. Every `IdolEngine` owns one of these and runs all its MLX work on it.
///
/// The thread lives until `stop()`: its loop holds the object, so dropping the last reference alone does
/// not end it. `IdolEngine` stops its thread in `deinit`; a host that owns an `MLXThread` directly calls
/// `stop()` itself.
public final class MLXThread: @unchecked Sendable {
    private let condition = NSCondition()
    private var jobs: [() -> Void] = []
    private var stopping = false
    private var thread: Thread!

    public init(name: String = "idol.mlx") {
        thread = Thread { [unowned self] in self.loop() }
        thread.name = name
        thread.qualityOfService = .userInitiated
        thread.stackSize = 16 << 20
        thread.start()
    }

    /// Queues `job`. Returns `false`, and drops the job, once `stop()` was called.
    @discardableResult
    public func async(_ job: @escaping () -> Void) -> Bool {
        condition.lock(); defer { condition.unlock() }
        guard !stopping else { return false }
        jobs.append(job); condition.signal()
        return true
    }

    /// Runs `body` on the MLX thread and returns its result to the caller's task.
    /// Throws `CancellationError` once `stop()` was called.
    public func run<T: Sendable>(_ body: @escaping () throws -> T) async throws -> T {
        try await withCheckedThrowingContinuation { continuation in
            let queued = async { continuation.resume(with: Result { try body() }) }
            if !queued { continuation.resume(throwing: CancellationError()) }
        }
    }

    /// Lets the thread finish the jobs already queued, then exit. Idempotent; returns at once.
    public func stop() {
        condition.lock(); stopping = true; condition.signal(); condition.unlock()
    }

    /// The thread has exited (after `stop()` and its last job).
    public var isFinished: Bool { thread.isFinished }

    private func loop() {
        while true {
            condition.lock()
            while jobs.isEmpty && !stopping { condition.wait() }
            if jobs.isEmpty { condition.unlock(); return }   // stopping, nothing left
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
