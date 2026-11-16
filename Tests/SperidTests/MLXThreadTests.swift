// MLXThreadTests.swift - the engine thread runs jobs in order and ends on stop() (tier 0)
// Copyright 2026 Vincent Gourbin

import Foundation
import Testing
@testable import IdolCore

/// Threads of this process (mach), to see an engine's thread come and go.
func processThreadCount() -> Int {
    var list: thread_act_array_t?
    var count: mach_msg_type_number_t = 0
    guard task_threads(mach_task_self_, &list, &count) == KERN_SUCCESS, let list else { return -1 }
    for i in 0..<Int(count) { mach_port_deallocate(mach_task_self_, list[i]) }
    vm_deallocate(mach_task_self_, vm_address_t(bitPattern: list), vm_size_t(Int(count) * MemoryLayout<thread_t>.stride))
    return Int(count)
}

func waitUntil(_ timeout: Duration = .seconds(5), _ condition: () -> Bool) async throws -> Bool {
    let end = ContinuousClock.now + timeout
    while ContinuousClock.now < end {
        if condition() { return true }
        try await Task.sleep(for: .milliseconds(20))
    }
    return condition()
}

@Suite("MLXThread", .serialized)
struct MLXThreadTests {
    @Test func runsJobsInOrderOnOneThread() async throws {
        let thread = MLXThread()
        defer { thread.stop() }
        let first = try await thread.run { Thread.current.name ?? "" }
        let values = try await thread.run { (0..<3).map { $0 * 2 } }
        #expect(first == "idol.mlx")
        #expect(values == [0, 2, 4])
    }

    @Test func stopDrainsTheQueueThenEnds() async throws {
        let thread = MLXThread()
        let log = StageLog()
        thread.async { Thread.sleep(forTimeInterval: 0.1); log.add(.preprocessing) }
        thread.async { log.add(.done) }
        thread.stop()
        #expect(try await waitUntil { thread.isFinished })
        #expect(log.all == [.preprocessing, .done])   // queued before stop: both ran
        #expect(!thread.async {})
        await #expect(throws: CancellationError.self) { _ = try await thread.run { 1 } }
    }

    @Test func dropsNoThreadAcrossCycles() async throws {
        let before = processThreadCount()
        for _ in 0..<20 {
            let thread = MLXThread()
            _ = try await thread.run { 0 }
            thread.stop()
            #expect(try await waitUntil { thread.isFinished })
        }
        #expect(try await waitUntil { processThreadCount() <= before + 2 })
    }
}
