# Worked solutions

One executed notebook per course notebook, with a worked solution to every exercise: code that computes the
answer, followed by an explanation of what the result means and, where the exercise was designed to be
surprising, why.

The solutions are generated from `build_solutions.py`, which takes the exercise texts from the course builder
(so they never drift apart): `python solutions/build_solutions.py` rebuilds and executes them all.

**For instructors:** if you use the exercises for assessed work, keep the solutions in a private repository
and remove the `solutions` glob from `.github/workflows/tests.yml`.
