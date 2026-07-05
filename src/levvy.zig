const std = @import("std");

const del_cost: u16 = 2;
const skip_cost: u16 = 2;
const sub_cost: u16 = 3;
const streak_bias: u16 = 3;

const case_setting = 1;

export fn fuzzy_search(query: [*:0]const u8, number_of_lines: c_uint, input: [*]const [*:0]const u8, output: [*]u16) callconv(.c) c_int {
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
