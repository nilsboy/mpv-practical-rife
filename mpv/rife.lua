-- rife.lua: mpv runtime toggle for Practical-RIFE frame interpolation.
--
-- Place in ~/.config/mpv/scripts/ and press the toggle key (default Ctrl+r)
-- while playing. Configuration comes from mpv options (script section [rife]
-- of ~/.config/mpv/mpv.conf or ~/.config/mpv/rife.conf) and MPV_RIFE_*
-- environment variables; see README.md.

local mp = require "mp"
local options = require "mp.options"

local o = {
    model = "",
    backend = "auto",
    device = "",
    threads = 0,
    factor_num = 2,
    factor_den = 1,
    sc = 0.15,
    pad = 128,
    matrix = "709",
    vs_threads = 2,
    vpy = "",
    script_path = "",
    buffered_frames = 4,
    concurrent_frames = 2,
    autostart = false,
}

options.read_options(o)

local b64chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"

-- mpv filter suboptions cannot contain '=', so emit base64 without padding
-- (config.py re-pads before decoding).
local function b64encode(data)
    local pad = (3 - #data % 3) % 3
    data = data .. string.rep("\0", pad)
    local out = {}
    for i = 1, #data, 3 do
        local a, b, c = data:byte(i, i + 2)
        local n = a * 65536 + b * 256 + c
        local q1 = math.floor(n / 262144) % 64
        local q2 = math.floor(n / 4096) % 64
        local q3 = math.floor(n / 64) % 64
        local q4 = n % 64
        out[#out + 1] = b64chars:sub(q1 + 1, q1 + 1)
        out[#out + 1] = b64chars:sub(q2 + 1, q2 + 1)
        out[#out + 1] = b64chars:sub(q3 + 1, q3 + 1)
        out[#out + 1] = b64chars:sub(q4 + 1, q4 + 1)
    end
    local result = table.concat(out)
    if pad == 2 then
        result = result:sub(1, -3)
    elseif pad == 1 then
        result = result:sub(1, -2)
    end
    return result
end

local function script_file()
    local name = o.vpy ~= "" and o.vpy or "rife_video.py"
    local dir = o.script_path ~= "" and o.script_path or (mp.get_script_directory() or "")
    if dir == "" then
        local snap = os.getenv("SNAP")
        if snap and snap ~= "" then
            dir = snap .. "/mpv"
        end
    end
    if dir == "" then
        return name
    end
    return dir .. "/" .. name
end

local function json_string()
    return string.format(
        '{"model":"%s","backend":"%s","device":"%s","threads":%s,' ..
        '"factor_num":%s,"factor_den":%s,"sc":%s,"pad":%s,' ..
        '"matrix":"%s","vs_threads":%s}',
        o.model, o.backend, o.device, tostring(o.threads or 0),
        tostring(o.factor_num), tostring(o.factor_den), tostring(o.sc), tostring(o.pad),
        o.matrix, tostring(o.vs_threads)
    )
end

local function filter_string()
    return string.format(
        "vapoursynth=%s:buffered-frames=%s:concurrent-frames=%s:user-data=%s",
        script_file(), tostring(o.buffered_frames), tostring(o.concurrent_frames),
        b64encode(json_string())
    )
end

local function describe()
    local factor = o.factor_num / o.factor_den
    local desc = string.format("%.2gx", factor)
    if o.backend ~= "auto" then
        desc = desc .. " " .. o.backend
    end
    return desc
end

local function is_active()
    local filters = mp.get_property_native("vf", {})
    for _, f in ipairs(filters) do
        if f.label == "rife" or (not f.label and f.name == "vapoursynth") then
            return true
        end
    end
    return false
end

local function osd(text)
    mp.osd_message(text, 2)
    mp.msg.info(text)
end

local function apply()
    if is_active() then
        return
    end
    mp.commandv("vf", "add", "@rife:" .. filter_string())
    osd("RIFE on (" .. describe() .. ")")
end

local function toggle()
    if is_active() then
        mp.commandv("vf", "remove", "@rife")
        osd("RIFE off")
    else
        apply()
    end
end

mp.add_key_binding("Ctrl+r", "rife-toggle", toggle)
mp.register_script_message("rife-toggle", toggle)

local function on_file_load()
    if o.autostart and not is_active() then
        apply()
    end
end

mp.register_event("file-loaded", on_file_load)
