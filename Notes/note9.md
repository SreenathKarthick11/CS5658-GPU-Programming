# Notes of GPU Programming

```text
Date : 3 Nov 2026
```
---

## Delauany Triangluation

Given a set of points $P = {p_1,p_2,...p_3}$ , the **Delaunany Triangulation** states that the circumcircle of every triangle contains no other input in its interior.


## Bowyer-Watson Algorithm

- Start with a large super triangle (contains all input points)
- Insert input points one at a time
- Find Bad Triangle
- The Cavity
- Find the Cavity Boundary
- Remove all non boundary edges
- Retriangulate the Cavity

> for details refer slides

---

