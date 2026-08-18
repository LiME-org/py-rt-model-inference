
# Discrete time: the smallest time unit is 1.
EPSILON = 1

# Various type aliases of `int` to express different kinds of time-related concepts.
Instant = int
Duration = int

ReleaseWindow = tuple[Instant, Instant]
