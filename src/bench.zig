// benchmark scalar fuzzy_search vs fuzzy_search_simd
//
// usage: zig build bench
//
// runs over two embedded real data files (from levvy-prototype) plus a
// synthetic set, so numbers are reproducible

const std = @import("std");
const levvy = @import("levvy.zig");

const SearchFn = *const fn ([*:0]const u8, c_uint, [*]const [*:0]const u8, [*]u16) callconv(.c) c_int;

const queries = [_][*:0]const u8{
    "awaitbun",
    "vimkeymapset",
    "refactoring",
    "constresult",
    "xq",
};

const Dataset = struct {
    name: []const u8,
    content: []const u8, // empty means synthetic
};

const datasets = [_]Dataset{
    .{ .name = "test3.txt (145 lines)", .content = @embedFile("bench-data/test3.txt") },
    .{ .name = "termtest.txt (10k lines)", .content = @embedFile("bench-data/termtest.txt") },
    .{ .name = "synthetic (2000 lines)", .content = "" },
};

fn now_ns() u64 {
    var ts: std.os.linux.timespec = undefined;
    _ = std.os.linux.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

fn bench_one(func: SearchFn, query: [*:0]const u8, lines: []const [*:0]const u8, out: [*]u16, reps: usize) u64 {
    var best: u64 = std.math.maxInt(u64);
    // warmup
    for (0..2) |_| _ = func(query, @intCast(lines.len), lines.ptr, out);
    for (0..reps) |_| {
        const start = now_ns();
        _ = func(query, @intCast(lines.len), lines.ptr, out);
        const t = now_ns() - start;
        if (t < best) best = t;
    }
    return best;
}

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    std.debug.print("simd width: {d} x u16\n", .{std.simd.suggestVectorLength(u16) orelse 8});

    var grand_scalar: u64 = 0;
    var grand_simd: u64 = 0;

    for (datasets) |dataset| {
        var lines: std.ArrayList([*:0]const u8) = .empty;

        if (dataset.content.len > 0) {
            var it = std.mem.splitScalar(u8, dataset.content, '\n');
            while (it.next()) |line| {
                try lines.append(allocator, try allocator.dupeZ(u8, line));
            }
        } else {
            var prng = std.Random.DefaultPrng.init(1234);
            const r = prng.random();
            const pool = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOP_./(){}=:;<> \"'0123456789";
            for (0..2000) |_| {
                const len = r.intRangeAtMost(usize, 10, 120);
                const buf = try allocator.allocSentinel(u8, len, 0);
                for (buf) |*c| c.* = pool[r.intRangeAtMost(usize, 0, pool.len - 1)];
                try lines.append(allocator, buf);
            }
        }

        const n = lines.items.len;
        var longest: usize = 0;
        for (lines.items) |l| longest = @max(longest, std.mem.len(l));

        std.debug.print("\n== {s}, longest line {d} ==\n", .{ dataset.name, longest });

        const out_scalar = try allocator.alloc(u16, n);
        const out_simd = try allocator.alloc(u16, n);

        const reps = 20;
        var total_scalar: u64 = 0;
        var total_simd: u64 = 0;

        for (queries) |query| {
            const t_scalar = bench_one(levvy.fuzzy_search, query, lines.items, out_scalar.ptr, reps);
            const t_simd = bench_one(levvy.fuzzy_search_simd, query, lines.items, out_simd.ptr, reps);
            total_scalar += t_scalar;
            total_simd += t_simd;

            var mismatches: usize = 0;
            for (out_scalar, out_simd) |a, b| {
                if (a != b) mismatches += 1;
            }

            const speedup = @as(f64, @floatFromInt(t_scalar)) / @as(f64, @floatFromInt(t_simd));
            std.debug.print("query {s:>14}: scalar {d:>8.3} ms, simd {d:>8.3} ms, speedup {d:.2}x{s}\n", .{
                query,
                @as(f64, @floatFromInt(t_scalar)) / 1e6,
                @as(f64, @floatFromInt(t_simd)) / 1e6,
                speedup,
                if (mismatches > 0) " !! OUTPUT MISMATCH" else "",
            });
            if (mismatches > 0) std.debug.print("  {d} mismatching lines\n", .{mismatches});
        }

        grand_scalar += total_scalar;
        grand_simd += total_simd;

        const total_speedup = @as(f64, @floatFromInt(total_scalar)) / @as(f64, @floatFromInt(total_simd));
        std.debug.print("dataset total: scalar {d:.3} ms, simd {d:.3} ms, speedup {d:.2}x\n", .{
            @as(f64, @floatFromInt(total_scalar)) / 1e6,
            @as(f64, @floatFromInt(total_simd)) / 1e6,
            total_speedup,
        });
    }

    const grand_speedup = @as(f64, @floatFromInt(grand_scalar)) / @as(f64, @floatFromInt(grand_simd));
    std.debug.print("\ngrand total: scalar {d:.3} ms, simd {d:.3} ms, speedup {d:.2}x\n", .{
        @as(f64, @floatFromInt(grand_scalar)) / 1e6,
        @as(f64, @floatFromInt(grand_simd)) / 1e6,
        grand_speedup,
    });
}
