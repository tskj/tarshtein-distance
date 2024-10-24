const std = @import("std");

// The code here is specific to Lua 5.1
// This has been tested with LuaJIT 5.1, specifically
pub const c = @cImport({
    @cInclude("luaconf.h");
    @cInclude("lua.h");
    @cInclude("lualib.h");
    @cInclude("lauxlib.h");
});

const LuaState = c.lua_State;
const FnReg = c.luaL_Reg;

/// A Zig function called by Lua must accept a single ?*LuaState parameter and must
/// return a c_int representing the number of return values pushed onto the stack
export fn fuzzy_search(lua: ?*LuaState) callconv(.C) c_int {
    const il = c.lua_tointeger(lua, 2);
    _ = il;
    const ql = c.lua_tointeger(lua, 4);
    _ = ql;

    c.lua_pushinteger(lua, 34);
    return 1;
}

const adder_reg: FnReg = .{ .name = "adder", .func = fuzzy_search };

const lib_fn_reg = [_]FnReg{ adder_reg, FnReg{} };

/// This is the entrypoint into the library from a Lua script
export fn luaopen_levvy(lua: ?*LuaState) callconv(.C) c_int {
    c.luaL_register(lua.?, "levvy", @ptrCast(&lib_fn_reg[0]));
    return 1;
}
