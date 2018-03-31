--[[

    How to use:

    local IKS = require 'IncrementalKolmogorovSmirnov'

    IKS:AddElement(OBSERVATION, SAMPLE_ID) -- -> void
    and
    IKS:RemoveElement(OBSERVATION, SAMPLE_ID) -- -> void

    OBSERVATION is a real number and SAMPLE_ID is either 1 or 2.

    local RejectNullHypotheis = IKS:Test() -- -> boolean

    Test assumes alpha = 0.001 and that |A| = |B| = m = n.
  ]]


local Treap = require 'support.Treap'

local idx = {}
local mt = {__index = idx}

function idx:KS()
  assert(self.n[1] == self.n[2])
  local n = self.n[1]
  if n == 0 then return 0 end
  return math.max(self.treap.max_value, -self.treap.min_value) / n
end

function idx:Kuiper()
  assert(self.n[1] == self.n[2])
  local n = self.n[1]
  if n == 0 then return 0 end
  return (self.treap.max_value - self.treap.min_value) / n
end

idx.V           = idx.KS
idx.Statistic   = idx.KS
idx.Stat        = idx.KS
idx.D           = idx.KS
idx.KSStat      = idx.KS
idx.KSStatistic = idx.KS
idx.PValue      = idx.KS

function idx:Test(ca)
  -- ca = ca or 1.22 -- alpha = 0.10
  -- ca = ca or 1.36 -- alpha = 0.05
  -- ca = ca or 1.48 -- alpha = 0.025
  -- ca = ca or 1.63 -- alpha = 0.01
  -- ca = ca or 1.73 -- alpha = 0.005
  ca = ca or 1.95 -- alpha = 0.001
  local n = self.n[1]
  return self:KS() > ca * math.sqrt(2 * n / n ^ 2), supremum
end

local meta_key = {
  __lt = function(a, b)
    if a[1] == b[1] then
      return a[2] < b[2]
    end
    return a[1] < b[1]
  end,
  __le = function(a, b)
    if a[1] == b[1] then
      return a[2] <= b[2]
    end
    return a[1] < b[1]
  end,
  __eq = function(a, b)
    return a[1] == b[1] and a[2] == b[2]
  end,
}

function idx:AddElement(key, group, deb)
  if #self.reuse > 0 then
    local k = self.reuse[#self.reuse].key
    k[1], k[2] = key, group
    key = k
  else
    key = setmetatable({key, group}, meta_key)
  end
  self.n[group] = self.n[group] + 1

  local left, left_g, right, val

  left, right = Treap.SplitKeepRight(self.treap, key)

  left, left_g = Treap.SplitGreatest(left)
  val = left_g and left_g.value or 0
  left = Treap.Merge(left, left_g)

  right = Treap.Merge(Treap.CreateNode(key, val, self.reuse[#self.reuse]), right)
  self.reuse[#self.reuse] = nil

  Treap.SumAll(right, group == 1 and 1 or -1)

  self.treap = Treap.Merge(left, right)
end

idx.Add = idx.AddElement

function idx:RemoveElement(key, group)
  self.rem_key[1], self.rem_key[2] = key, group
  local key = self.rem_key
  self.n[group] = self.n[group] - 1
  local left, right, right_l

  left, right = Treap.SplitKeepRight(self.treap, key)
  right_l, right = Treap.SplitSmallest(right)

  if right_l and right_l.key == key then
    Treap.SumAll(right, group == 1 and -1 or 1)
    self.reuse[#self.reuse + 1] = right_l
  else
    right = Treap.Merge(right_l, right)
  end

  self.treap = Treap.Merge(left, right)
end

idx.Remove = idx.RemoveElement

return function()
  return setmetatable({
      treap = nil,
      n = {0, 0},
      reuse = {},
      rem_key = setmetatable({0, 0}, meta_key)
    }, mt)
end