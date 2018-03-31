return function(t)
  local mu = 0
  local m2 = 0
  for i, v in ipairs(t) do
    mu = mu + v / #t
    m2 = m2 + v ^ 2 / #t
  end
  return mu, math.sqrt(m2 - mu ^ 2), m2
end
