local function MergeTables(...)
  local nargs = select('#', ...)
  if nargs == 1 then
    return MergeTables(unpack(select(1, ...)))
  end
  local t = {}
  for i = 1, nargs do
    local s = select(i, ...)
    for j, v in ipairs(s) do
      table.insert(t, v)
    end
  end
  return t
end

return MergeTables