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

fn bench_handle(handle: ?*anyopaque, query: [*:0]const u8, out: [*]u16, threads: c_uint, reps: usize) u64 {
    var best: u64 = std.math.maxInt(u64);
    // warmup
    for (0..2) |_| _ = levvy.levvy_search(handle, query, out, threads);
    for (0..reps) |_| {
        const start = now_ns();
        _ = levvy.levvy_search(handle, query, out, threads);
        const t = now_ns() - start;
        if (t < best) best = t;
    }
    return best;
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
    var grand_scan: u64 = 0;
    var grand_cache: u64 = 0;
    var grand_mt: u64 = 0;

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
        const out_scan = try allocator.alloc(u16, n);
        const out_cache = try allocator.alloc(u16, n);
        const out_mt = try allocator.alloc(u16, n);

        const create_start = now_ns();
        const handle = levvy.levvy_create(lines.items.ptr, @intCast(n));
        const create_ns = now_ns() - create_start;
        defer levvy.levvy_destroy(handle);
        std.debug.print("levvy_create (one-time preprocessing): {d:.3} ms\n", .{@as(f64, @floatFromInt(create_ns)) / 1e6});

        const reps = 20;
        var total_scalar: u64 = 0;
        var total_simd: u64 = 0;
        var total_scan: u64 = 0;
        var total_cache: u64 = 0;
        var total_mt: u64 = 0;

        for (queries) |query| {
            const t_scalar = bench_one(levvy.fuzzy_search, query, lines.items, out_scalar.ptr, reps);
            const t_simd = bench_one(levvy.fuzzy_search_simd, query, lines.items, out_simd.ptr, reps);
            const t_scan = bench_one(levvy.fuzzy_search_simd_scan, query, lines.items, out_scan.ptr, reps);
            const t_cache = bench_handle(handle, query, out_cache.ptr, 1, reps);
            const t_mt = bench_handle(handle, query, out_mt.ptr, 0, reps);
            total_scalar += t_scalar;
            total_simd += t_simd;
            total_scan += t_scan;
            total_cache += t_cache;
            total_mt += t_mt;

            var mismatches: usize = 0;
            for (out_scalar, out_simd) |a, b| {
                if (a != b) mismatches += 1;
            }
            for (out_scalar, out_scan) |a, b| {
                if (a != b) mismatches += 1;
            }
            for (out_scalar, out_cache) |a, b| {
                if (a != b) mismatches += 1;
            }
            for (out_scalar, out_mt) |a, b| {
                if (a != b) mismatches += 1;
            }

            std.debug.print("query {s:>14}: scalar {d:>8.3} | simd {d:>7.3} ({d:>5.2}x) | scan {d:>7.3} ({d:>5.2}x) | cache {d:>7.3} ({d:>5.2}x) | +mt {d:>7.3} ({d:>5.2}x){s}\n", .{
                query,
                @as(f64, @floatFromInt(t_scalar)) / 1e6,
                @as(f64, @floatFromInt(t_simd)) / 1e6,
                @as(f64, @floatFromInt(t_scalar)) / @as(f64, @floatFromInt(t_simd)),
                @as(f64, @floatFromInt(t_scan)) / 1e6,
                @as(f64, @floatFromInt(t_scalar)) / @as(f64, @floatFromInt(t_scan)),
                @as(f64, @floatFromInt(t_cache)) / 1e6,
                @as(f64, @floatFromInt(t_scalar)) / @as(f64, @floatFromInt(t_cache)),
                @as(f64, @floatFromInt(t_mt)) / 1e6,
                @as(f64, @floatFromInt(t_scalar)) / @as(f64, @floatFromInt(t_mt)),
                if (mismatches > 0) " !! OUTPUT MISMATCH" else "",
            });
            if (mismatches > 0) std.debug.print("  {d} mismatching lines\n", .{mismatches});
        }

        grand_scalar += total_scalar;
        grand_simd += total_simd;
        grand_scan += total_scan;
        grand_cache += total_cache;
        grand_mt += total_mt;

        std.debug.print("dataset total: scalar {d:.3} | simd {d:.3} ({d:.2}x) | scan {d:.3} ({d:.2}x) | cache {d:.3} ({d:.2}x) | +mt {d:.3} ({d:.2}x)\n", .{
            @as(f64, @floatFromInt(total_scalar)) / 1e6,
            @as(f64, @floatFromInt(total_simd)) / 1e6,
            @as(f64, @floatFromInt(total_scalar)) / @as(f64, @floatFromInt(total_simd)),
            @as(f64, @floatFromInt(total_scan)) / 1e6,
            @as(f64, @floatFromInt(total_scalar)) / @as(f64, @floatFromInt(total_scan)),
            @as(f64, @floatFromInt(total_cache)) / 1e6,
            @as(f64, @floatFromInt(total_scalar)) / @as(f64, @floatFromInt(total_cache)),
            @as(f64, @floatFromInt(total_mt)) / 1e6,
            @as(f64, @floatFromInt(total_scalar)) / @as(f64, @floatFromInt(total_mt)),
        });
    }

    std.debug.print("\ngrand total: scalar {d:.3} | simd {d:.3} ({d:.2}x) | scan {d:.3} ({d:.2}x) | cache {d:.3} ({d:.2}x) | +mt {d:.3} ({d:.2}x)\n", .{
        @as(f64, @floatFromInt(grand_scalar)) / 1e6,
        @as(f64, @floatFromInt(grand_simd)) / 1e6,
        @as(f64, @floatFromInt(grand_scalar)) / @as(f64, @floatFromInt(grand_simd)),
        @as(f64, @floatFromInt(grand_scan)) / 1e6,
        @as(f64, @floatFromInt(grand_scalar)) / @as(f64, @floatFromInt(grand_scan)),
        @as(f64, @floatFromInt(grand_cache)) / 1e6,
        @as(f64, @floatFromInt(grand_scalar)) / @as(f64, @floatFromInt(grand_cache)),
        @as(f64, @floatFromInt(grand_mt)) / 1e6,
        @as(f64, @floatFromInt(grand_scalar)) / @as(f64, @floatFromInt(grand_mt)),
    });
}
