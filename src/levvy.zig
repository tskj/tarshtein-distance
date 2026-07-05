const std = @import("std");

const del_cost: u16 = 2;
const skip_cost: u16 = 2;
const sub_cost: u16 = 3;
const streak_bias: u16 = 3;

const case_setting = 1;

pub export fn fuzzy_search(query: [*:0]const u8, number_of_lines: c_uint, input: [*]const [*:0]const u8, output: [*]u16) callconv(.c) c_int {
    const q_len: u16 = @as(u16, @intCast(std.mem.len(query)));

    var longest_line_length: u16 = 0;
    for (input, 0..number_of_lines) |line, index| {
        const l_len = @as(u16, @intCast(std.mem.len(line)));
        output[index] = l_len;
        if (l_len > longest_line_length) longest_line_length = l_len;
    }

    // two rows are needed at a time
    // and two tables, one for streak and one for non-streak
    const memory_requirement = (longest_line_length + 1) * 2 * 2;

    const allocator = std.heap.page_allocator;
    const buffer = allocator.alloc(u16, memory_requirement) catch {
        return -1;
    };
    defer allocator.free(buffer);

    const half = memory_requirement / 2;
    const dp_curr = buffer[0..half];
    const dp_prev = buffer[half..];

    var shortest_dist: ?u16 = null;
    for (0..number_of_lines) |i| {
        const d = compute_distance(query, q_len, input[i], output[i], longest_line_length - output[i], dp_curr, dp_prev);
        output[i] = d;
        if (shortest_dist == null or d < shortest_dist.?) shortest_dist = d;
    }

    if (shortest_dist == null) return -1;
    return @as(c_int, @intCast(shortest_dist.?));
}

fn compute_distance(q: [*]const u8, q_len: u16, h: [*]const u8, h_len: u16, padding: u16, dp_current: []u16, dp_previous: []u16) u16 {
    var dp_curr = dp_current;
    var dp_prev = dp_previous;

    const H: u16 = h_len + 1;
    const B: u16 = 2;
    const padding_cost: u16 = padding * skip_cost;
    const bias: u16 = @min(q_len, h_len) * streak_bias; // stops distance going negative

    // base case
    var h_i: u16 = 0;
    while (h_i < H) : (h_i += 1) {
        const dist: u16 = (h_len - h_i) * skip_cost + padding_cost + bias;
        dp_prev[h_i * B + 0] = dist;
        dp_prev[h_i * B + 1] = dist;
    }

    var q_i = q_len;
    while (q_i > 0) {
        q_i -= 1;

        const dist = (q_len - q_i) * del_cost + padding_cost + bias;
        dp_curr[h_len * B + 0] = dist;
        dp_curr[h_len * B + 1] = dist;

        h_i = h_len;
        while (h_i > 0) {
            h_i -= 1;

            var a = q[q_i];
            var b = h[h_i];

            if (case_setting == 2) {
                if (65 <= a and a <= 90) a |= 32;
                if (65 <= b and b <= 90) b |= 32;
            } else if (case_setting == 1 and 97 <= a and a <= 122) {
                if (65 <= b and b <= 90) b |= 32;
            }

            const is_match = a == b;

            const index_current = h_i * B;
            const index_next = (h_i + 1) * B;

            const del_cost_total = del_cost + dp_prev[index_current + 0];
            const skip_cost_total = skip_cost + dp_curr[index_next + 0];

            const match_cost =
                if (is_match) dp_prev[index_next + 1] else sub_cost + dp_prev[index_next + 0];

            dp_curr[index_current] = @min(del_cost_total, @min(skip_cost_total, match_cost));

            const del_cost_cm1 = del_cost + dp_prev[index_current + 1];
            const skip_cost_cm1 = skip_cost + dp_curr[index_next];

            const match_cost_cm1 =
                if (is_match) dp_prev[index_next + 1] - streak_bias else sub_cost + dp_prev[index_next];

            dp_curr[index_current + 1] = @min(del_cost_cm1, @min(skip_cost_cm1, match_cost_cm1));
        }

        const tmp = dp_curr;
        dp_curr = dp_prev;
        dp_prev = tmp;
    }

    // if we could have a bias we need to remove one
    // because the first character can't start in a streak
    return dp_prev[0] - if (bias > 0) streak_bias else 0;
}

// -- simd implementation --
//
// per cell, every candidate is "an already-computed dp value + a constant":
//   cm=0: prev0[h_i] + del | curr0[h_i+1] + skip | prev1[h_i+1] + 0  or prev0[h_i+1] + sub
//   cm=1: prev1[h_i] + del | curr0[h_i+1] + skip | prev1[h_i+1] - streak or prev0[h_i+1] + sub
//
// everything except skip depends only on the previous row, so those lanes are
// computed for simd_width cells at a time (match vs sub blended on the
// is_match predicate, saturating u16 arithmetic). the skip candidate reads the
// cell just written in the same row -- that loop-carried dependency is folded
// in afterwards as a cheap scalar right-to-left min-scan (exact, this is the
// same escape hatch striped smith-waterman implementations use).
//
// the dp rows are stored as four separate planes (prev0/prev1/curr0/curr1)
// rather than interleaved pairs so the vector loads are contiguous.

const simd_width = std.simd.suggestVectorLength(u16) orelse 8;

pub export fn fuzzy_search_simd(query: [*:0]const u8, number_of_lines: c_uint, input: [*]const [*:0]const u8, output: [*]u16) callconv(.c) c_int {
    const q_len: u16 = @as(u16, @intCast(std.mem.len(query)));

    var longest_line_length: u16 = 0;
    for (input, 0..number_of_lines) |line, index| {
        const l_len = @as(u16, @intCast(std.mem.len(line)));
        output[index] = l_len;
        if (l_len > longest_line_length) longest_line_length = l_len;
    }

    const allocator = std.heap.page_allocator;

    // four dp planes with simd_width slack so vector loads/stores may run past h_len
    const plane_len: usize = @as(usize, longest_line_length) + 1 + simd_width;
    const planes = allocator.alloc(u16, plane_len * 4) catch return -1;
    defer allocator.free(planes);
    @memset(planes, 0xffff);

    // padded copies of the line so vector loads never read past the caller's
    // allocation; lowercased once per line instead of per cell
    const scratch_len: usize = @as(usize, longest_line_length) + simd_width;
    const h_scratch = allocator.alloc(u8, scratch_len * 2) catch return -1;
    defer allocator.free(h_scratch);

    var shortest_dist: ?u16 = null;
    for (0..number_of_lines) |i| {
        const h_len = output[i];
        const h_exact = h_scratch[0 .. h_len + simd_width];
        const h_lower = h_scratch[scratch_len..][0 .. h_len + simd_width];
        for (0..h_len) |j| {
            const c = input[i][j];
            h_exact[j] = c;
            h_lower[j] = if ('A' <= c and c <= 'Z') c | 32 else c;
        }
        @memset(h_exact[h_len..], 0);
        @memset(h_lower[h_len..], 0);

        const d = compute_distance_simd(
            query,
            q_len,
            h_exact,
            h_lower,
            h_len,
            longest_line_length - h_len,
            planes[0 * plane_len ..][0..plane_len],
            planes[1 * plane_len ..][0..plane_len],
            planes[2 * plane_len ..][0..plane_len],
            planes[3 * plane_len ..][0..plane_len],
        );
        output[i] = d;
        if (shortest_dist == null or d < shortest_dist.?) shortest_dist = d;
    }

    if (shortest_dist == null) return -1;
    return @as(c_int, @intCast(shortest_dist.?));
}

fn compute_distance_simd(
    q: [*]const u8,
    q_len: u16,
    h_exact: []const u8,
    h_lower: []const u8,
    h_len: u16,
    padding: u16,
    prev0_in: []u16,
    prev1_in: []u16,
    curr0_in: []u16,
    curr1_in: []u16,
) u16 {
    const W = simd_width;

    var prev0 = prev0_in;
    var prev1 = prev1_in;
    var curr0 = curr0_in;
    var curr1 = curr1_in;

    const padding_cost: u16 = padding * skip_cost;
    const bias: u16 = @min(q_len, h_len) * streak_bias; // stops distance going negative

    // base case
    var h_i: u16 = 0;
    while (h_i <= h_len) : (h_i += 1) {
        const dist: u16 = (h_len - h_i) * skip_cost + padding_cost + bias;
        prev0[h_i] = dist;
        prev1[h_i] = dist;
    }

    const delv: @Vector(W, u16) = @splat(del_cost);
    const subv: @Vector(W, u16) = @splat(sub_cost);
    const streakv: @Vector(W, u16) = @splat(streak_bias);

    var q_i = q_len;
    while (q_i > 0) {
        q_i -= 1;

        var a = q[q_i];
        var h_used = h_exact;
        if (case_setting == 2) {
            if ('A' <= a and a <= 'Z') a |= 32;
            h_used = h_lower;
        } else if (case_setting == 1 and 'a' <= a and a <= 'z') {
            h_used = h_lower;
        }
        const av: @Vector(W, u8) = @splat(a);

        // pass 1: del and match/sub candidates for W cells at a time
        // (saturating ops so the slack lanes beyond h_len can't wrap)
        var j: u16 = 0;
        while (j < h_len) : (j += W) {
            const hv: @Vector(W, u8) = h_used[j..][0..W].*;
            const is_match = hv == av;

            const p0_curr: @Vector(W, u16) = prev0[j..][0..W].*;
            const p1_curr: @Vector(W, u16) = prev1[j..][0..W].*;
            const p0_next: @Vector(W, u16) = prev0[j + 1 ..][0..W].*;
            const p1_next: @Vector(W, u16) = prev1[j + 1 ..][0..W].*;

            const match_or_sub_0 = @select(u16, is_match, p1_next, p0_next +| subv);
            const match_or_sub_1 = @select(u16, is_match, p1_next -| streakv, p0_next +| subv);

            curr0[j..][0..W].* = @min(p0_curr +| delv, match_or_sub_0);
            curr1[j..][0..W].* = @min(p1_curr +| delv, match_or_sub_1);
        }

        // boundary cell (whole query deleted); set after pass 1 since the
        // last vector store may have spilled past h_len
        const boundary: u16 = (q_len - q_i) * del_cost + padding_cost + bias;
        curr0[h_len] = boundary;
        curr1[h_len] = boundary;

        // pass 2: fold in the skip candidate right to left; skip resets the
        // streak so both cm rows fold from the (already folded) cm=0 chain
        h_i = h_len;
        while (h_i > 0) {
            h_i -= 1;
            // branchy on purpose: the skip candidate only rarely wins, so the
            // predicted-not-taken branch beats unconditional @min stores here
            // (measured ~2.4x vs ~2.1x total speedup)
            const skip_total = skip_cost + curr0[h_i + 1];
            if (skip_total < curr0[h_i]) curr0[h_i] = skip_total;
            if (skip_total < curr1[h_i]) curr1[h_i] = skip_total;
        }

        std.mem.swap([]u16, &prev0, &curr0);
        std.mem.swap([]u16, &prev1, &curr1);
    }

    // if we could have a bias we need to remove one
    // because the first character can't start in a streak
    return prev0[0] - if (bias > 0) streak_bias else 0;
}

test "simple test" {
    var list: std.ArrayList(i32) = .empty;
    defer list.deinit(std.testing.allocator); // try commenting this out and see if zig detects the memory leak!
    try list.append(std.testing.allocator, 42);
    try std.testing.expectEqual(@as(i32, 42), list.pop().?);
}

// -- property tests (mirroring property.test.ts in levvy-prototype) --

const max_test_len = 100;

fn test_distance(q: []const u8, h: []const u8, padding: u16) u16 {
    var curr: [(max_test_len + 1) * 2]u16 = undefined;
    var prev: [(max_test_len + 1) * 2]u16 = undefined;
    return compute_distance(q.ptr, @intCast(q.len), h.ptr, @intCast(h.len), padding, curr[0..], prev[0..]);
}

fn random_string(r: std.Random, buf: []u8, max_len: usize) []u8 {
    const pool = "abcdefgXYZAbC_./(){}=:;<> \"'0123456789";
    const len = r.intRangeAtMost(usize, 0, max_len);
    for (buf[0..len]) |*c| c.* = pool[r.intRangeAtMost(usize, 0, pool.len - 1)];
    return buf[0..len];
}

test "property: when q == h and padding: 0, distance should be 0" {
    var prng = std.Random.DefaultPrng.init(42);
    const r = prng.random();
    var buf: [max_test_len]u8 = undefined;
    for (0..300) |_| {
        const s = random_string(r, &buf, 60);
        try std.testing.expectEqual(@as(u16, 0), test_distance(s, s, 0));
    }
}

test "property: empty haystack costs q.len * del_cost; empty query costs h.len * skip_cost" {
    var prng = std.Random.DefaultPrng.init(43);
    const r = prng.random();
    var buf: [max_test_len]u8 = undefined;
    for (0..200) |_| {
        const s = random_string(r, &buf, 60);
        try std.testing.expectEqual(@as(u16, @intCast(s.len * del_cost)), test_distance(s, "", 0));
        try std.testing.expectEqual(@as(u16, @intCast(s.len * skip_cost)), test_distance("", s, 0));
    }
}

test "property: smart case: an all-lowercase query is case-insensitive in the haystack" {
    var prng = std.Random.DefaultPrng.init(44);
    const r = prng.random();
    const lower_pool = "abcdefghij_./ 0123";
    var qbuf: [16]u8 = undefined;
    var hbuf: [32]u8 = undefined;
    var hbuf_recased: [32]u8 = undefined;
    for (0..200) |_| {
        const q_len = r.intRangeAtMost(usize, 0, 8);
        for (qbuf[0..q_len]) |*c| c.* = lower_pool[r.intRangeAtMost(usize, 0, lower_pool.len - 1)];
        const h_len = r.intRangeAtMost(usize, 0, 12);
        for (hbuf[0..h_len], hbuf_recased[0..h_len]) |*c, *rc| {
            c.* = lower_pool[r.intRangeAtMost(usize, 0, lower_pool.len - 1)];
            rc.* = if (r.boolean() and 'a' <= c.* and c.* <= 'z') c.* - 32 else c.*;
        }
        try std.testing.expectEqual(
            test_distance(qbuf[0..q_len], hbuf[0..h_len], 0),
            test_distance(qbuf[0..q_len], hbuf_recased[0..h_len], 0),
        );
    }
}

test "property: contiguous matches beat scattered matches" {
    var prng = std.Random.DefaultPrng.init(45);
    const r = prng.random();
    const letters = "abcdefghijklmnop";
    var qbuf: [8]u8 = undefined;
    var contiguous: [max_test_len]u8 = undefined;
    var scattered: [max_test_len]u8 = undefined;
    for (0..200) |_| {
        const q_len = r.intRangeAtMost(usize, 2, 6);
        for (qbuf[0..q_len]) |*c| c.* = letters[r.intRangeAtMost(usize, 0, letters.len - 1)];
        const q = qbuf[0..q_len];
        const extra = r.intRangeAtMost(usize, 0, 4);
        const gaps = q_len - 1 + extra;
        const h_len = q_len + gaps;

        @memset(contiguous[0..h_len], '_');
        @memcpy(contiguous[0..q_len], q);

        @memset(scattered[0..h_len], '_');
        for (q, 0..) |c, i| scattered[i * 2] = c;

        try std.testing.expect(test_distance(q, contiguous[0..h_len], 0) < test_distance(q, scattered[0..h_len], 0));
    }
}

fn test_distance_simd(q: []const u8, h: []const u8, padding: u16) u16 {
    const W = simd_width;
    var h_exact: [max_test_len + W]u8 = undefined;
    var h_lower: [max_test_len + W]u8 = undefined;
    for (h, 0..) |c, j| {
        h_exact[j] = c;
        h_lower[j] = if ('A' <= c and c <= 'Z') c | 32 else c;
    }
    @memset(h_exact[h.len..], 0);
    @memset(h_lower[h.len..], 0);

    const plane_len = max_test_len + 1 + W;
    var planes: [4][plane_len]u16 = undefined;
    for (&planes) |*p| @memset(p, 0xffff);

    return compute_distance_simd(
        q.ptr,
        @intCast(q.len),
        h_exact[0 .. h.len + W],
        h_lower[0 .. h.len + W],
        @intCast(h.len),
        padding,
        &planes[0],
        &planes[1],
        &planes[2],
        &planes[3],
    );
}

test "simd: agrees with scalar on random inputs" {
    var prng = std.Random.DefaultPrng.init(47);
    const r = prng.random();
    var qbuf: [max_test_len]u8 = undefined;
    var hbuf: [max_test_len]u8 = undefined;
    for (0..1000) |_| {
        const q = random_string(r, &qbuf, 20);
        const h = random_string(r, &hbuf, 90);
        const padding = r.intRangeAtMost(u16, 0, 8);
        try std.testing.expectEqual(test_distance(q, h, padding), test_distance_simd(q, h, padding));
    }
}

test "simd: fuzzy_search_simd matches fuzzy_search" {
    const query: [*:0]const u8 = "hello";
    const number_of_lines: c_uint = 6;
    const input_strings = [_][*:0]const u8{
        "hello",
        "world",
        "hell",
        "help",
        "hel",
        "hel__",
    };
    var outputs = [_]u16{ 0, 0, 0, 0, 0, 0 };
    var outputs_simd = [_]u16{ 0, 0, 0, 0, 0, 0 };

    const result = fuzzy_search(query, number_of_lines, input_strings[0..].ptr, &outputs);
    const result_simd = fuzzy_search_simd(query, number_of_lines, input_strings[0..].ptr, &outputs_simd);

    try std.testing.expectEqual(result, result_simd);
    try std.testing.expectEqualSlices(u16, outputs[0..], outputs_simd[0..]);
}

test "property: padding adds exactly padding * skip_cost" {
    var prng = std.Random.DefaultPrng.init(46);
    const r = prng.random();
    var qbuf: [max_test_len]u8 = undefined;
    var hbuf: [max_test_len]u8 = undefined;
    for (0..200) |_| {
        const q = random_string(r, &qbuf, 20);
        const h = random_string(r, &hbuf, 40);
        const padding = r.intRangeAtMost(u16, 0, 8);
        try std.testing.expectEqual(
            test_distance(q, h, 0) + padding * skip_cost,
            test_distance(q, h, padding),
        );
    }
}

test "fuzzy_search simple test" {
    const query: [*:0]const u8 = "hello";
    const number_of_lines: c_uint = 6;
    const input_strings = [_][*:0]const u8{
        "hello",
        "world",
        "hell",
        "help",
        "hel",
        "hel__",
    };
    var outputs = [_]u16{ 0, 0, 0, 0, 0, 0 };

    // Call the fuzzy_search function with the test data
    const result = fuzzy_search(query, number_of_lines, input_strings[0..].ptr, &outputs);

    // Ensure the function did not return an error
    try std.testing.expect(result >= 0);

    // Check that the shortest distance is as expected
    try std.testing.expectEqual(@as(c_int, 0), result);

    // Validate the distances computed for each input string
    try std.testing.expectEqual(@as(u16, 0), outputs[0]); // "hello" vs "hello"
    try std.testing.expect(outputs[1] > 0); // "hello" vs "world"
    try std.testing.expect(outputs[2] > 0); // "hello" vs "hell"
    try std.testing.expect(outputs[3] > 0); // "hello" vs "help"

    // Optionally, print the distances for manual verification
    std.debug.print("Shortest distance: {d}\n", .{result});
    for (0..number_of_lines) |index| {
        const input = input_strings[index][0..std.mem.len(input_strings[index])];
        std.debug.print("Distance between '{s}' and '{s}': {d}\n", .{ query[0..std.mem.len(query)], input, outputs[index] });
    }
}
