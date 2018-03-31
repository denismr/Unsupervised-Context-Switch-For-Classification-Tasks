local idx = {}
local mt = {__index = idx}

local function SquaredEuclideanDistance(a, b, features)
  local d = 0
  if not features then
    for i = 1, #a - 1 do
      local A = a[i]
      local B = b[i]
      d = d + (A - B) ^ 2
    end
  else
    for _, f in ipairs(features) do
      local A = a[f]
      local B = b[f]
      d = d + (A - B) ^ 2
    end
  end
  return d
end

function idx:LOOT() -- Leave One Out Table
  local prev_ii = self.ignore_identicals
  self.ignore_identicals = true

  local tab = {}
  for i, evt in ipairs(self.data) do
    tab[i] = select(2, self:Classify(evt))
  end

  self.ignore_identicals = prev_ii
  return tab
end

function idx:LOO() -- Leave One Out Table
  local prev_ii = self.ignore_identicals
  self.ignore_identicals = true

  local predicted = {}
  local correct = {}
  local distance = {}
  for i, evt in ipairs(self.data) do
    predicted[i], distance[i] = self:Classify(evt)
    correct[i] = evt[self.target_feature]
  end

  return predicted, correct, distance
end

function idx:Classify(x)
  local closest_d = math.huge
  local closest_l = 0
  local check_identicals = not self.ignore_identicals
  for i, v in ipairs(self.data) do
    if x ~= v or check_identicals then
      local d = self.distance_function(v, x, self.features)
      if d < closest_d then
        closest_d = d
        closest_l = v[self.target_feature]
      end
    end
  end
  return closest_l, closest_d
end

function idx:SetReference(data)
  self.data = data
end

mt.__call = idx.Classify


return function(settings)
  if settings then
    return setmetatable({
      data = settings.data,
      features = settings.features,
      distance_function = settings.distance_function or settings.distance or SquaredEuclideanDistance,
      target_feature = settings.target or settings.target_feature or #data[1],
      ignore_identicals = settings.ignore_identicals == nil and true or settings.ignore_identicals,
    }, mt)
  else
    return setmetatable({}, mt)
  end
end