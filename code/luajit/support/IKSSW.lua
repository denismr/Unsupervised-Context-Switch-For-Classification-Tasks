-- IKS Sliding Window
local IKS = require 'support.IKS'

local index = {}
local metatable = {__index = index}

function index:Add(value)
  local to_remove = self.sw[self.nex]
  if to_remove then
    self.iks:RemoveElement(to_remove, 2)
  end
  self.sw[self.nex] = value
  self.iks:AddElement(value, 2)
  self.nex = self.nex + 1
  if self.nex > self.n then
    self.nex = 1
  end

  if #self.sw == self.n then
    return self.iks:Test(self.ca)
  else
    return false
  end
end

function index:Increment(value)
  local to_remove = self.sw[self.nex]
  if to_remove then
    self.iks:RemoveElement(to_remove, 2)
  end
  self.sw[self.nex] = value
  self.iks:AddElement(value, 2)
  self.nex = self.nex + 1
  if self.nex > self.n then
    self.nex = 1
  end
end

function index:Computable()
  return #self.sw == self.n
end

function index:PValue()
  return self.iks:PValue()
end

function index:KS()
  return self.iks:KS()
end

function index:Kuiper()
  return self.iks:Kuiper()
end

function index:A()
  return {unpack(self._A)}
end

function index:B()
  return {unpack(self.sw)}
end

function index:UpdateFixedSet(fixed_set)
  self._A = fixed_set
  self.n = #fixed_set
  self.iks = IKS()
  for i, v in ipairs(self._A) do
    self.iks:AddElement(v, 1)
  end
  for i, v in ipairs(self.sw) do
    self.iks:AddElement(v, 2)
  end
end


return function(fixed_set, ca, tru_ca)
  local iks = IKS()
  local n = #fixed_set
  for i, v in ipairs(fixed_set) do
    iks:AddElement(v, 1)
  end
  
  local self = setmetatable({
    ca = ca,
    iks = iks,
    n = n,
    sw = {},
    nex = 1,
    _A = fixed_set,
  }, metatable)

  if type(ca == 'boolean') then
    if ca == true then
      for i, v in ipairs(fixed_set) do
        self:Increment(v)
      end
    end
    ca = tru_ca
  elseif type(ca == 'table') then
    for i, v in ipairs(ca) do
      self:Increment(v)
    end
    ca = tru_ca
  end

  return self
end