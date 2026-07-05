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

const ComputeFn = fn ([*]const u8, u16, []const u8, []const u8, u16, u16, []u16, []u16, []u16, []u16) u16;

fn search_with_planes(comptime compute: ComputeFn, query: [*:0]const u8, number_of_lines: c_uint, input: [*]const [*:0]const u8, output: [*]u16) c_int {
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

        // the planes are reused across lines: a previous, longer line leaves
        // real (small) values in the lanes past this line's h_len, which the
        // computations may read -- re-poison the reachable slack
        const wipe_start = @as(usize, h_len) + 1;
        for (0..4) |p| {
            const plane = planes[p * plane_len ..][0..plane_len];
            @memset(plane[wipe_start..@min(plane_len, wipe_start + simd_width)], 0xffff);
        }

        const d = compute(
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

pub export fn fuzzy_search_simd(query: [*:0]const u8, number_of_lines: c_uint, input: [*]const [*:0]const u8, output: [*]u16) callconv(.c) c_int {
    return search_with_planes(compute_distance_simd, query, number_of_lines, input, output);
}

pub export fn fuzzy_search_simd_scan(query: [*:0]const u8, number_of_lines: c_uint, input: [*]const [*:0]const u8, output: [*]u16) callconv(.c) c_int {
    return search_with_planes(compute_distance_simd_scan, query, number_of_lines, input, output);
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

// -- simd + scan implementation --
//
// same candidate computation as compute_distance_simd, but the skip fold is
// restructured away: unrolling the recurrence
//   curr0[h] = min(cand0[h], skip + curr0[h+1])
// gives
//   curr0[h] = min over m >= h of (cand0[m] + (m - h) * skip)
// which is a suffix min-plus scan with linear drift. that scan is computed in
// log2(W) shuffle+add+min steps per W-lane block, so the loop-carried
// dependency shrinks from one-per-cell to one carry value per block.

// shift lanes towards index 0 by `step`, vacated high lanes take fill[0]
fn shiftLow(v: @Vector(simd_width, u16), comptime step: comptime_int, fill: @Vector(simd_width, u16)) @Vector(simd_width, u16) {
    const mask = comptime blk: {
        var m: [simd_width]i32 = undefined;
        for (&m, 0..) |*x, k| x.* = if (k + step < simd_width) @as(i32, @intCast(k + step)) else -1;
        break :blk m;
    };
    return @shuffle(u16, v, fill, mask);
}

fn compute_distance_simd_scan(
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
    const V = @Vector(W, u16);

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

    const sentinel: V = @splat(0xffff);
    const delv: V = @splat(del_cost);
    const subv: V = @splat(sub_cost);
    const streakv: V = @splat(streak_bias);
    const skipv: V = @splat(skip_cost);
    // cross-block skip drift: lane k is (block_end - k) skips away from the carry
    const ramp: V = comptime blk: {
        var r: [W]u16 = undefined;
        for (&r, 0..) |*x, k| x.* = @intCast((W - k) * skip_cost);
        break :blk r;
    };

    const nblocks = (@as(usize, h_len) + W - 1) / W;

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

        const boundary: u16 = (q_len - q_i) * del_cost + padding_cost + bias;

        // carry holds s[block_end] of the block to the right. for the
        // rightmost block that's either exactly cell h_len (h_len % W == 0,
        // whole query deleted from here: boundary) or a lane past it
        // (sentinel; cell h_len then sits inside the block, where its del
        // candidate prev0[h_len] + del reproduces the boundary exactly)
        var carry: u16 = if (h_len % W == 0) boundary else 0xffff;

        var b = nblocks;
        while (b > 0) {
            b -= 1;
            const j = b * W;

            const hv: @Vector(W, u8) = h_used[j..][0..W].*;
            const is_match = hv == av;

            const p0_curr: V = prev0[j..][0..W].*;
            const p1_curr: V = prev1[j..][0..W].*;
            const p0_next: V = prev0[j + 1 ..][0..W].*;
            const p1_next: V = prev1[j + 1 ..][0..W].*;

            const cand0 = @min(p0_curr +| delv, @select(u16, is_match, p1_next, p0_next +| subv));
            const cand1 = @min(p1_curr +| delv, @select(u16, is_match, p1_next -| streakv, p0_next +| subv));

            // fold the cross-block carry, then suffix-scan within the block;
            // s is then exactly curr0 (skip candidate fully included)
            var s = @min(cand0, @as(V, @splat(carry)) +| ramp);
            comptime var step = 1;
            inline while (step < W) : (step *= 2) {
                s = @min(s, shiftLow(s, step, sentinel) +| @as(V, @splat(step * skip_cost)));
            }

            curr0[j..][0..W].* = s;

            // curr1[h] = min(cand1[h], skip + curr0[h+1]); the lane shifted in
            // at the top is s[block_end], i.e. the incoming carry
            const s_next = shiftLow(s, 1, @as(V, @splat(carry)));
            curr1[j..][0..W].* = @min(cand1, s_next +| skipv);

            carry = s[0];
        }

        // boundary cell (whole query deleted); also repairs vector-store spill
        curr0[h_len] = boundary;
        curr1[h_len] = boundary;

        std.mem.swap([]u16, &prev0, &curr0);
        std.mem.swap([]u16, &prev1, &curr1);
    }

    // if we could have a bias we need to remove one
    // because the first character can't start in a streak
    return prev0[0] - if (bias > 0) streak_bias else 0;
}

// -- persistent handle api --
//
// a picker re-queries the same candidate set on every keystroke; the
// per-call setup (strlen, padded exact+lowercase copies, dp plane
// allocation) is the dominant cost for short queries. levvy_create does all
// of that once; levvy_search then only runs the dp kernel, optionally
// spread across threads (lines are independent).
//
//   handle = levvy_create(lines, n)      -- preprocess, returns null on oom
//   levvy_search(handle, q, out, threads) -- threads: 0 = one per cpu
//   levvy_destroy(handle)

const max_supported_threads = 64;
const is_linux = @import("builtin").os.tag == .linux;

// a parallel search only pays if every thread gets a decent slice of work;
// roughly the number of dp cells a pool wakeup is worth
const cells_per_thread = 100_000;

const Levvy = struct {
    n: usize,
    longest: u16,
    total_chars: usize,
    plane_len: usize,
    max_threads: usize,
    line_lens: []u16,
    line_offsets: []usize, // into the exact/lower arenas
    exact: []u8,
    lower: []u8,
    planes: []u16, // max_threads * 4 * plane_len

    // persistent worker pool (linux): workers park on a futex over
    // `generation`; a search publishes `job`, bumps the generation and wakes
    // everyone; workers signal completion by decrementing `pending`.
    // thread spawn costs ~100us+ under wsl2, a futex wakeup is ~microseconds,
    // which is what makes threading pay at keystroke granularity.
    generation: std.atomic.Value(u32),
    pending: std.atomic.Value(u32),
    pool_stop: bool,
    job: Job,
    nworkers: usize,
    workers_tried: bool,
    workers: [max_supported_threads]std.Thread,

    const Job = struct {
        query: [*:0]const u8 = undefined,
        q_len: u16 = 0,
        output: [*]u16 = undefined,
        want: usize = 0,
        chunk: usize = 0,
    };
};

fn futex_wait(addr: *const std.atomic.Value(u32), expect: u32) void {
    _ = std.os.linux.futex_4arg(addr, .{ .cmd = .WAIT, .private = true }, expect, null);
}

fn futex_wake(addr: *const std.atomic.Value(u32), count: u32) void {
    _ = std.os.linux.futex_3arg(addr, .{ .cmd = .WAKE, .private = true }, count);
}

fn worker_loop(self: *Levvy, index: usize) void {
    var seen: u32 = 0;
    while (true) {
        while (true) {
            const g = self.generation.load(.acquire);
            if (g != seen) {
                seen = g;
                break;
            }
            futex_wait(&self.generation, g);
        }
        if (self.pool_stop) return;

        const job = self.job;
        // main runs chunk 0, worker `index` runs chunk index + 1
        const start = (index + 1) * job.chunk;
        if (index + 1 < job.want and start < self.n) {
            const end = @min(start + job.chunk, self.n);
            score_range(self, job.query, job.q_len, job.output, index + 1, start, end);
        }

        if (self.pending.fetchSub(1, .release) == 1) {
            futex_wake(&self.pending, 1);
        }
    }
}

fn ensure_workers(self: *Levvy) void {
    if (self.workers_tried) return;
    self.workers_tried = true;
    const target = self.max_threads - 1; // the calling thread participates too
    for (0..target) |i| {
        self.workers[i] = std.Thread.spawn(.{}, worker_loop, .{ self, i }) catch break;
        self.nworkers = i + 1;
    }
}

const levvy_allocator = std.heap.page_allocator;

fn levvy_create_inner(input: [*]const [*:0]const u8, number_of_lines: c_uint) !*Levvy {
    const n: usize = number_of_lines;
    const self = try levvy_allocator.create(Levvy);
    errdefer levvy_allocator.destroy(self);

    self.n = n;

    self.line_lens = try levvy_allocator.alloc(u16, n);
    errdefer levvy_allocator.free(self.line_lens);
    self.line_offsets = try levvy_allocator.alloc(usize, n);
    errdefer levvy_allocator.free(self.line_offsets);

    var longest: u16 = 0;
    var total: usize = 0;
    var total_chars: usize = 0;
    for (0..n) |i| {
        const l_len = @as(u16, @intCast(std.mem.len(input[i])));
        self.line_lens[i] = l_len;
        self.line_offsets[i] = total;
        total += @as(usize, l_len) + simd_width;
        total_chars += l_len;
        if (l_len > longest) longest = l_len;
    }
    self.longest = longest;
    self.total_chars = total_chars;

    self.exact = try levvy_allocator.alloc(u8, total);
    errdefer levvy_allocator.free(self.exact);
    self.lower = try levvy_allocator.alloc(u8, total);
    errdefer levvy_allocator.free(self.lower);

    for (0..n) |i| {
        const off = self.line_offsets[i];
        const l_len = self.line_lens[i];
        for (0..l_len) |j| {
            const c = input[i][j];
            self.exact[off + j] = c;
            self.lower[off + j] = if ('A' <= c and c <= 'Z') c | 32 else c;
        }
        @memset(self.exact[off + l_len ..][0..simd_width], 0);
        @memset(self.lower[off + l_len ..][0..simd_width], 0);
    }

    self.plane_len = @as(usize, longest) + 1 + simd_width;
    self.max_threads = @max(@min(std.Thread.getCpuCount() catch 1, max_supported_threads), 1);
    self.planes = try levvy_allocator.alloc(u16, self.max_threads * 4 * self.plane_len);
    errdefer levvy_allocator.free(self.planes);

    self.generation = std.atomic.Value(u32).init(0);
    self.pending = std.atomic.Value(u32).init(0);
    self.pool_stop = false;
    self.job = .{};
    self.nworkers = 0;
    self.workers_tried = false;

    return self;
}

pub export fn levvy_create(input: [*]const [*:0]const u8, number_of_lines: c_uint) callconv(.c) ?*anyopaque {
    const self = levvy_create_inner(input, number_of_lines) catch return null;
    return @ptrCast(self);
}

pub export fn levvy_destroy(handle: ?*anyopaque) callconv(.c) void {
    const self: *Levvy = @ptrCast(@alignCast(handle orelse return));
    if (self.nworkers > 0) {
        self.pool_stop = true;
        _ = self.generation.fetchAdd(1, .release);
        futex_wake(&self.generation, @intCast(self.nworkers));
        for (self.workers[0..self.nworkers]) |t| t.join();
    }
    levvy_allocator.free(self.planes);
    levvy_allocator.free(self.lower);
    levvy_allocator.free(self.exact);
    levvy_allocator.free(self.line_offsets);
    levvy_allocator.free(self.line_lens);
    levvy_allocator.destroy(self);
}

fn score_range(self: *const Levvy, query: [*:0]const u8, q_len: u16, output: [*]u16, thread_index: usize, start: usize, end: usize) void {
    const plane_len = self.plane_len;
    const planes = self.planes[thread_index * 4 * plane_len ..][0 .. 4 * plane_len];
    @memset(planes, 0xffff);

    for (start..end) |i| {
        const h_len = self.line_lens[i];
        const off = self.line_offsets[i];

        // re-poison the slack lanes a previous longer line may have dirtied
        const wipe_start = @as(usize, h_len) + 1;
        for (0..4) |p| {
            const plane = planes[p * plane_len ..][0..plane_len];
            @memset(plane[wipe_start..@min(plane_len, wipe_start + simd_width)], 0xffff);
        }

        output[i] = compute_distance_simd_scan(
            query,
            q_len,
            self.exact[off..][0 .. @as(usize, h_len) + simd_width],
            self.lower[off..][0 .. @as(usize, h_len) + simd_width],
            h_len,
            self.longest - h_len,
            planes[0 * plane_len ..][0..plane_len],
            planes[1 * plane_len ..][0..plane_len],
            planes[2 * plane_len ..][0..plane_len],
            planes[3 * plane_len ..][0..plane_len],
        );
    }
}

pub export fn levvy_search(handle: ?*anyopaque, query: [*:0]const u8, output: [*]u16, threads: c_uint) callconv(.c) c_int {
    const self: *Levvy = @ptrCast(@alignCast(handle orelse return -1));
    if (self.n == 0) return -1;

    const q_len: u16 = @as(u16, @intCast(std.mem.len(query)));

    // size the crew to the work: below ~cells_per_thread dp cells per thread
    // the coordination costs more than it buys
    const cells = @as(usize, q_len) * self.total_chars;
    const worth = @max(cells / cells_per_thread, 1);
    var want: usize = if (threads == 0) @min(worth, self.max_threads) else @min(threads, self.max_threads);
    want = @min(want, @max(self.n / 64, 1));

    if (want > 1 and is_linux) {
        ensure_workers(self);
        want = @min(want, self.nworkers + 1);
    }

    if (want <= 1) {
        score_range(self, query, q_len, output, 0, 0, self.n);
    } else if (is_linux) {
        // dispatch to the parked pool
        const chunk = (self.n + want - 1) / want;
        self.job = .{ .query = query, .q_len = q_len, .output = output, .want = want, .chunk = chunk };
        self.pending.store(@intCast(self.nworkers), .monotonic);
        _ = self.generation.fetchAdd(1, .release);
        // wake count is an `int` in the kernel: must be positive, not maxInt(u32)
        futex_wake(&self.generation, @intCast(self.nworkers));

        score_range(self, query, q_len, output, 0, 0, @min(chunk, self.n));

        while (true) {
            const p = self.pending.load(.acquire);
            if (p == 0) break;
            futex_wait(&self.pending, p);
        }
    } else {
        // no futex here: spawn per call (fine for one-shot use)
        const chunk = (self.n + want - 1) / want;
        var handles: [max_supported_threads]?std.Thread = @splat(null);
        for (1..want) |t| {
            const start = t * chunk;
            if (start >= self.n) break;
            const end = @min(start + chunk, self.n);
            handles[t] = std.Thread.spawn(.{}, score_range, .{ self, query, q_len, output, t, start, end }) catch blk: {
                // couldn't spawn: do that slice on this thread instead
                score_range(self, query, q_len, output, t, start, end);
                break :blk null;
            };
        }
        score_range(self, query, q_len, output, 0, 0, @min(chunk, self.n));
        for (handles) |h| if (h) |thread| thread.join();
    }

    var shortest: u16 = output[0];
    for (1..self.n) |i| {
        if (output[i] < shortest) shortest = output[i];
    }
    return @as(c_int, @intCast(shortest));
}

// -- single-line scorer for streaming sorters (telescope et al) --
//
// telescope scores entries one at a time as they stream in, so the batch
// handle api doesn't fit; this entry point scores a single line with
// zero allocation (thread-local scratch). semantics tailored to pickers:
//
//   - returns -1 when the query is not a smart-case subsequence of the
//     line ("doesn't match", the caller should discard the entry) --
//     note a non-match stays a non-match for any extension of the query,
//     which is exactly the invariant telescope's discard-caching assumes
//   - otherwise returns the levvy distance with the line padded to
//     `pad_to` columns, so scores are length-normalized and comparable
//     across lines without knowing the longest line up front

const score_max_line = 2048;
const score_plane_len = score_max_line + 1 + simd_width;

threadlocal var score_planes: [4 * score_plane_len]u16 = undefined;
threadlocal var score_planes_ready: bool = false;
threadlocal var score_exact: [score_max_line + simd_width]u8 = undefined;
threadlocal var score_lower: [score_max_line + simd_width]u8 = undefined;

// scores a single line against the query. no match/no-match gate: every
// line gets a real levvy distance and the caller ranks by it (a poor match
// simply gets a large distance and sinks). this is the whole point of the
// algorithm -- a one-character typo is a cheap substitution, not a rejection,
// so it still ranks near the top rather than disappearing.
pub export fn levvy_score(query: [*:0]const u8, line: [*:0]const u8, pad_to: c_uint) callconv(.c) c_int {
    const q_len_full = std.mem.len(query);
    const h_len_full = std.mem.len(line);

    // very long lines are scored on their prefix
    const h_len: u16 = @intCast(@min(h_len_full, score_max_line));
    const q_len: u16 = @intCast(@min(q_len_full, score_max_line));

    if (!score_planes_ready) {
        @memset(&score_planes, 0xffff);
        score_planes_ready = true;
    }

    // same slack hygiene as the batch drivers: previous (longer) lines
    // leave real values past this line's h_len
    const wipe_start = @as(usize, h_len) + 1;
    for (0..4) |p| {
        const plane = score_planes[p * score_plane_len ..][0..score_plane_len];
        @memset(plane[wipe_start..@min(score_plane_len, wipe_start + simd_width)], 0xffff);
    }

    for (0..h_len) |j| {
        const c = line[j];
        score_exact[j] = c;
        score_lower[j] = if ('A' <= c and c <= 'Z') c | 32 else c;
    }
    @memset(score_exact[h_len..][0..simd_width], 0);
    @memset(score_lower[h_len..][0..simd_width], 0);

    const padding: u16 = if (pad_to > h_len) @intCast(@min(pad_to - h_len, 4096)) else 0;

    const d = compute_distance_simd_scan(
        query,
        q_len,
        score_exact[0 .. @as(usize, h_len) + simd_width],
        score_lower[0 .. @as(usize, h_len) + simd_width],
        h_len,
        padding,
        score_planes[0 * score_plane_len ..][0..score_plane_len],
        score_planes[1 * score_plane_len ..][0..score_plane_len],
        score_planes[2 * score_plane_len ..][0..score_plane_len],
        score_planes[3 * score_plane_len ..][0..score_plane_len],
    );
    return @as(c_int, @intCast(d));
}

// -- match position reconstruction (for highlighting) --
//
// walks an optimal path through the full dp table and reports which line
// positions were consumed by 'match' operations. only called for entries
// actually displayed, so a plain scalar full table is plenty.
//
// note the full table uses the same min(q,h)*streak_bias offset as the fast
// implementations: the (min-1) variant the typescript reference uses can
// dip one streak_bias below zero on the top-left cm=1 cell, which u16
// can't represent. the offset is uniform across cells, so path choices are
// unaffected.

fn adjusted_chars(q_c: u8, h_c: u8) [2]u8 {
    var a = q_c;
    var b = h_c;
    if (case_setting == 2) {
        if ('A' <= a and a <= 'Z') a |= 32;
        if ('A' <= b and b <= 'Z') b |= 32;
    } else if (case_setting == 1 and 'a' <= a and a <= 'z') {
        if ('A' <= b and b <= 'Z') b |= 32;
    }
    return .{ a, b };
}

// dp must be (q_len + 1) * (h_len + 1) * 2 long
fn compute_full_table(q: [*]const u8, q_len: u16, h: [*]const u8, h_len: u16, dp: []u16) void {
    const B: usize = 2;
    const BH: usize = B * (@as(usize, h_len) + 1);
    const bias: u16 = @min(q_len, h_len) * streak_bias;

    var q_i: usize = 0;
    while (q_i <= q_len) : (q_i += 1) {
        const dist: u16 = @as(u16, @intCast(q_len - q_i)) * del_cost + bias;
        dp[q_i * BH + h_len * B + 0] = dist;
        dp[q_i * BH + h_len * B + 1] = dist;
    }
    var h_i: usize = 0;
    while (h_i <= h_len) : (h_i += 1) {
        const dist: u16 = @as(u16, @intCast(h_len - h_i)) * skip_cost + bias;
        dp[@as(usize, q_len) * BH + h_i * B + 0] = dist;
        dp[@as(usize, q_len) * BH + h_i * B + 1] = dist;
    }

    q_i = q_len;
    while (q_i > 0) {
        q_i -= 1;
        h_i = h_len;
        while (h_i > 0) {
            h_i -= 1;
            const pair = adjusted_chars(q[q_i], h[h_i]);
            const is_match = pair[0] == pair[1];

            const del_total = del_cost + dp[(q_i + 1) * BH + h_i * B + 0];
            const skip_total = skip_cost + dp[q_i * BH + (h_i + 1) * B + 0];
            const match_total =
                if (is_match) dp[(q_i + 1) * BH + (h_i + 1) * B + 1] else sub_cost + dp[(q_i + 1) * BH + (h_i + 1) * B + 0];
            dp[q_i * BH + h_i * B + 0] = @min(del_total, @min(skip_total, match_total));

            const del_cm1 = del_cost + dp[(q_i + 1) * BH + h_i * B + 1];
            const skip_cm1 = skip_cost + dp[q_i * BH + (h_i + 1) * B + 0];
            const match_cm1 =
                if (is_match) dp[(q_i + 1) * BH + (h_i + 1) * B + 1] - streak_bias else sub_cost + dp[(q_i + 1) * BH + (h_i + 1) * B + 0];
            dp[q_i * BH + h_i * B + 1] = @min(del_cm1, @min(skip_cm1, match_cm1));
        }
    }
}

// walks the table from (0, 0), preferring match > substitute > delete > skip
// among cost-consistent options (mirrors path() in the prototype); returns
// how many match positions were written to out
fn walk_positions(q: [*]const u8, q_len: u16, h: [*]const u8, h_len: u16, dp: []const u16, out: [*]u16, out_cap: usize) usize {
    const B: usize = 2;
    const BH: usize = B * (@as(usize, h_len) + 1);

    var count: usize = 0;
    var q_i: usize = 0;
    var h_i: usize = 0;
    var cm: usize = 0;

    while (q_i < q_len or h_i < h_len) {
        const current: i32 = dp[q_i * BH + h_i * B + cm];

        var is_match = false;
        if (q_i < q_len and h_i < h_len) {
            const pair = adjusted_chars(q[q_i], h[h_i]);
            is_match = pair[0] == pair[1];
        }

        // match (or substitute)
        if (q_i < q_len and h_i < h_len) {
            if (is_match) {
                const op_cost: i32 = if (cm == 1) -@as(i32, streak_bias) else 0;
                if (op_cost + @as(i32, dp[(q_i + 1) * BH + (h_i + 1) * B + 1]) == current) {
                    if (count < out_cap) out[count] = @intCast(h_i);
                    count += 1;
                    q_i += 1;
                    h_i += 1;
                    cm = 1;
                    continue;
                }
            } else {
                if (sub_cost + @as(i32, dp[(q_i + 1) * BH + (h_i + 1) * B + 0]) == current) {
                    q_i += 1;
                    h_i += 1;
                    cm = 0;
                    continue;
                }
            }
        }

        // delete
        if (q_i < q_len and del_cost + @as(i32, dp[(q_i + 1) * BH + h_i * B + cm]) == current) {
            q_i += 1;
            continue;
        }

        // skip
        if (h_i < h_len and skip_cost + @as(i32, dp[q_i * BH + (h_i + 1) * B + 0]) == current) {
            h_i += 1;
            cm = 0;
            continue;
        }

        // no cost-consistent operation: table and walk disagree, give up
        // rather than loop forever (should be impossible)
        return count;
    }

    return count;
}

// returns the number of match positions written to out (0-based byte
// offsets into line, strictly increasing, at most min(out_cap, #query)).
// no gate: highlights whatever the optimal path actually matched, which
// may be nothing (returns 0) for an unrelated line -- consistent with
// levvy_score scoring every line rather than rejecting.
pub export fn levvy_positions(query: [*:0]const u8, line: [*:0]const u8, out: [*]u16, out_cap: c_uint) callconv(.c) c_int {
    const q_len_full = std.mem.len(query);
    const h_len_full = std.mem.len(line);

    const q_len: u16 = @intCast(@min(q_len_full, score_max_line));
    const h_len: u16 = @intCast(@min(h_len_full, score_max_line));
    if (q_len == 0) return 0;

    const allocator = std.heap.page_allocator;
    const dp = allocator.alloc(u16, (@as(usize, q_len) + 1) * (@as(usize, h_len) + 1) * 2) catch return -1;
    defer allocator.free(dp);

    compute_full_table(query, q_len, line, h_len, dp);
    const count = walk_positions(query, q_len, line, h_len, dp, out, out_cap);
    return @intCast(@min(count, out_cap));
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

fn test_distance_simd_scan(q: []const u8, h: []const u8, padding: u16) u16 {
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

    return compute_distance_simd_scan(
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

test "simd scan: agrees with scalar on random inputs" {
    var prng = std.Random.DefaultPrng.init(48);
    const r = prng.random();
    var qbuf: [max_test_len]u8 = undefined;
    var hbuf: [max_test_len]u8 = undefined;
    for (0..1000) |_| {
        const q = random_string(r, &qbuf, 20);
        const h = random_string(r, &hbuf, 90);
        const padding = r.intRangeAtMost(u16, 0, 8);
        try std.testing.expectEqual(test_distance(q, h, padding), test_distance_simd_scan(q, h, padding));
    }
}

test "simd scan: exact block-boundary line lengths" {
    // h_len % simd_width == 0 takes the carry-seeded path instead of the
    // in-block boundary identity; exercise both plus the empty line
    var prng = std.Random.DefaultPrng.init(49);
    const r = prng.random();
    var qbuf: [max_test_len]u8 = undefined;
    var hbuf: [max_test_len]u8 = undefined;
    const lens = [_]usize{ 0, 1, simd_width - 1, simd_width, simd_width + 1, 2 * simd_width, 4 * simd_width };
    for (0..100) |_| {
        const q = random_string(r, &qbuf, 12);
        for (lens) |h_len| {
            const pool = "abcABC_ ";
            for (hbuf[0..h_len]) |*c| c.* = pool[r.intRangeAtMost(usize, 0, pool.len - 1)];
            const h = hbuf[0..h_len];
            try std.testing.expectEqual(test_distance(q, h, 3), test_distance_simd_scan(q, h, 3));
        }
    }
}

test "exports agree on random multi-line input (plane reuse across lines)" {
    // long lines followed by short ones leave stale values in the dp planes;
    // this drives the three exported entry points over the same varied file
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(50);
    const r = prng.random();
    const pool = "abcdefgXYZAbC_./(){}=:;<> \"'0123456789";

    const n = 60;
    var lines: [n][*:0]const u8 = undefined;
    var allocated: [n][:0]u8 = undefined;
    for (0..n) |i| {
        // alternate long and short lines to stress stale-lane hygiene
        const max_len: usize = if (i % 2 == 0) 90 else 7;
        const len = r.intRangeAtMost(usize, 0, max_len);
        const buf = allocator.allocSentinel(u8, len, 0) catch unreachable;
        for (buf) |*c| c.* = pool[r.intRangeAtMost(usize, 0, pool.len - 1)];
        allocated[i] = buf;
        lines[i] = buf.ptr;
    }
    defer for (allocated) |buf| allocator.free(buf);

    const test_queries = [_][*:0]const u8{ "", "abc", "AbC", "xq0", "conxtime" };
    for (test_queries) |query| {
        var out_scalar: [n]u16 = undefined;
        var out_simd: [n]u16 = undefined;
        var out_scan: [n]u16 = undefined;
        const r_scalar = fuzzy_search(query, n, &lines, &out_scalar);
        const r_simd = fuzzy_search_simd(query, n, &lines, &out_simd);
        const r_scan = fuzzy_search_simd_scan(query, n, &lines, &out_scan);
        try std.testing.expectEqual(r_scalar, r_simd);
        try std.testing.expectEqual(r_scalar, r_scan);
        try std.testing.expectEqualSlices(u16, &out_scalar, &out_simd);
        try std.testing.expectEqualSlices(u16, &out_scalar, &out_scan);
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

test "handle api: agrees with fuzzy_search, reused across queries and thread counts" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(51);
    const r = prng.random();
    const pool = "abcdefgXYZAbC_./(){}=:;<> \"'0123456789";

    const n = 500;
    var lines: [n][*:0]const u8 = undefined;
    var allocated: [n][:0]u8 = undefined;
    for (0..n) |i| {
        const max_len: usize = if (i % 3 == 0) 90 else 9;
        const len = r.intRangeAtMost(usize, 0, max_len);
        const buf = allocator.allocSentinel(u8, len, 0) catch unreachable;
        for (buf) |*c| c.* = pool[r.intRangeAtMost(usize, 0, pool.len - 1)];
        allocated[i] = buf;
        lines[i] = buf.ptr;
    }
    defer for (allocated) |buf| allocator.free(buf);

    const handle = levvy_create(&lines, n);
    try std.testing.expect(handle != null);
    defer levvy_destroy(handle);

    const test_queries = [_][*:0]const u8{ "", "abc", "AbC", "xq0", "conxtime", "abc" };
    for (test_queries) |query| {
        var out_ref: [n]u16 = undefined;
        var out_single: [n]u16 = undefined;
        var out_auto: [n]u16 = undefined;
        var out_many: [n]u16 = undefined;

        const r_ref = fuzzy_search(query, n, &lines, &out_ref);
        const r_single = levvy_search(handle, query, &out_single, 1);
        const r_auto = levvy_search(handle, query, &out_auto, 0);
        const r_many = levvy_search(handle, query, &out_many, 3);

        try std.testing.expectEqual(r_ref, r_single);
        try std.testing.expectEqual(r_ref, r_auto);
        try std.testing.expectEqual(r_ref, r_many);
        try std.testing.expectEqualSlices(u16, &out_ref, &out_single);
        try std.testing.expectEqualSlices(u16, &out_ref, &out_auto);
        try std.testing.expectEqualSlices(u16, &out_ref, &out_many);
    }
}

test "levvy_score: always equals the batch distance (no gate, every line scored)" {
    var prng = std.Random.DefaultPrng.init(52);
    const r = prng.random();
    var hbuf: [max_test_len:0]u8 = undefined;
    var qbuf: [max_test_len:0]u8 = undefined;
    const pool = "abcDEfgH_./ 12";
    const pad_to: u16 = 128;

    for (0..1000) |_| {
        const h_len = r.intRangeAtMost(usize, 0, 60);
        for (hbuf[0..h_len]) |*c| c.* = pool[r.intRangeAtMost(usize, 0, pool.len - 1)];
        hbuf[h_len] = 0;

        // subsequence half the time, random junk (typos, unrelated) the rest
        var q_len: usize = 0;
        if (r.boolean()) {
            var h_i: usize = 0;
            while (h_i < h_len) : (h_i += 1) {
                if (r.boolean() and q_len < 12) {
                    qbuf[q_len] = hbuf[h_i];
                    q_len += 1;
                }
            }
        } else {
            q_len = r.intRangeAtMost(usize, 0, 10);
            for (qbuf[0..q_len]) |*c| c.* = pool[r.intRangeAtMost(usize, 0, pool.len - 1)];
        }
        qbuf[q_len] = 0;

        const got = levvy_score(qbuf[0..q_len :0], hbuf[0..h_len :0], pad_to);
        const padding: u16 = @intCast(pad_to - @min(h_len, pad_to));
        // every line, subsequence or not, gets exactly the batch distance
        try std.testing.expectEqual(
            @as(c_int, @intCast(test_distance(qbuf[0..q_len], hbuf[0..h_len], padding))),
            got,
        );
    }
}

test "levvy_score: ranking sanity for a picker (typos rank near, not rejected)" {
    // contiguous filename match beats scattered path match
    const q = "keymaps";
    const good = levvy_score(q, "plugin/keymaps.lua", 256);
    const scattered = levvy_score(q, "plugin/custom/key-map-something.lua", 256);
    const unrelated = levvy_score(q, "plugin/colorscheme.lua", 256);
    try std.testing.expect(good >= 0);
    try std.testing.expect(good < scattered);
    // an unrelated file isn't rejected -- it just scores worse and sinks
    try std.testing.expect(unrelated >= 0);
    try std.testing.expect(good < unrelated);

    // the motivating case: a one-character typo (grayvendel vs graywendel)
    // must score close to the exact match, and far better than an unrelated
    // file -- a substitution, not a rejection
    const exact = levvy_score("graywendel", "graywendel_migration.sql", 256);
    const typo = levvy_score("grayvendel", "graywendel_migration.sql", 256);
    const nope = levvy_score("grayvendel", "plugin/colorscheme.lua", 256);
    try std.testing.expect(typo >= 0);
    try std.testing.expect(typo > exact); // the typo costs something...
    try std.testing.expect(typo < exact + 5 * sub_cost); // ...but not much
    try std.testing.expect(typo < nope); // and ranks well above an unrelated file

    // smart case: lowercase query matches uppercase line
    try std.testing.expect(levvy_score("readme", "README.md", 256) >= 0);
}

test "positions: exact substring highlights the contiguous range" {
    var out: [16]u16 = undefined;
    const n = levvy_positions("abc", "xxabcxx", &out, out.len);
    try std.testing.expectEqual(@as(c_int, 3), n);
    try std.testing.expectEqualSlices(u16, &.{ 2, 3, 4 }, out[0..3]);

    // smart case
    const n2 = levvy_positions("keymaps", "plugin/KeyMaps.lua", &out, out.len);
    try std.testing.expectEqual(@as(c_int, 7), n2);
    try std.testing.expectEqualSlices(u16, &.{ 7, 8, 9, 10, 11, 12, 13 }, out[0..7]);

    // unrelated line: nothing matched, so zero highlight positions (not -1)
    try std.testing.expectEqual(@as(c_int, 0), levvy_positions("zzz", "abc", &out, out.len));
    // empty query
    try std.testing.expectEqual(@as(c_int, 0), levvy_positions("", "abc", &out, out.len));
}

test "positions: faraway characters still get matched (skips are paid either way)" {
    // every line character is consumed regardless, so matching a query char
    // (free) always beats deleting it and skipping the char anyway -- for a
    // subsequence query, every query character ends up highlighted
    var line: [64:0]u8 = undefined;
    @memset(line[0..64], '_');
    line[0] = 'a';
    line[63] = 'b';
    line[64] = 0;
    var out: [16]u16 = undefined;
    const n = levvy_positions("ab", &line, &out, out.len);
    try std.testing.expectEqual(@as(c_int, 2), n);
    try std.testing.expectEqualSlices(u16, &.{ 0, 63 }, out[0..2]);
}

test "positions: full table agrees with the scalar distance" {
    var prng = std.Random.DefaultPrng.init(53);
    const r = prng.random();
    var qbuf: [max_test_len]u8 = undefined;
    var hbuf: [max_test_len]u8 = undefined;
    var dp: [(20 + 1) * (max_test_len + 1) * 2]u16 = undefined;
    for (0..300) |_| {
        const q = random_string(r, &qbuf, 20);
        const h = random_string(r, &hbuf, 90);
        const q_len: u16 = @intCast(q.len);
        const h_len: u16 = @intCast(h.len);
        compute_full_table(q.ptr, q_len, h.ptr, h_len, dp[0 .. (@as(usize, q_len) + 1) * (@as(usize, h_len) + 1) * 2]);
        const bias: u16 = @min(q_len, h_len) * streak_bias;
        const from_table = dp[0] - if (bias > 0) streak_bias else 0;
        try std.testing.expectEqual(test_distance(q, h, 0), from_table);
    }
}

test "positions: walk invariants on random subsequence queries" {
    var prng = std.Random.DefaultPrng.init(54);
    const r = prng.random();
    var hbuf: [max_test_len:0]u8 = undefined;
    var qbuf: [max_test_len:0]u8 = undefined;
    var out: [32]u16 = undefined;
    const pool = "abcDEfgH_./ 12";

    for (0..300) |_| {
        const h_len = r.intRangeAtMost(usize, 1, 60);
        for (hbuf[0..h_len]) |*c| c.* = pool[r.intRangeAtMost(usize, 0, pool.len - 1)];
        hbuf[h_len] = 0;

        // draw a real subsequence as the query
        var q_len: usize = 0;
        var h_i: usize = 0;
        while (h_i < h_len) : (h_i += 1) {
            if (r.boolean() and q_len < 12) {
                qbuf[q_len] = hbuf[h_i];
                q_len += 1;
            }
        }
        qbuf[q_len] = 0;

        const n = levvy_positions(qbuf[0..q_len :0], hbuf[0..h_len :0], &out, out.len);
        try std.testing.expect(n >= 0);
        const count: usize = @intCast(n);
        try std.testing.expect(count <= q_len);
        for (out[0..count], 0..) |p, k| {
            try std.testing.expect(p < h_len);
            if (k > 0) try std.testing.expect(out[k - 1] < p);
        }
        // the highlighted characters must be matchable against the query in
        // order (the walk may delete query characters in between)
        var qi: usize = 0;
        for (out[0..count]) |p| {
            var found = false;
            while (qi < q_len) : (qi += 1) {
                const pair = adjusted_chars(qbuf[qi], hbuf[p]);
                if (pair[0] == pair[1]) {
                    qi += 1;
                    found = true;
                    break;
                }
            }
            try std.testing.expect(found);
        }
    }
}

test "handle api: empty input and null handle" {
    var out: [1]u16 = undefined;
    const lines: [1][*:0]const u8 = .{"x"};
    const handle = levvy_create(&lines, 0);
    defer levvy_destroy(handle);
    try std.testing.expectEqual(@as(c_int, -1), levvy_search(handle, "q", &out, 0));
    try std.testing.expectEqual(@as(c_int, -1), levvy_search(null, "q", &out, 0));
    levvy_destroy(null);
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
