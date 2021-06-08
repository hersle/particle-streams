##  Overview

* The simulation area is divided into a grid, such that one particle is in at most 4 cells at any given time.
* Every cell remembers every particle it contains, and every particle remembers which cells it is in.
* Only nearby particles from adjacent cells are collision checked, allowing real-time simulation of 10000s of particles. 

## Usage instructions

1. Install [Julia](https://julialang.org/).
2. Run `julia`, hit `]` to enter the package manager, then `add GLMakie` (for real-time visualization) and `add StaticArrays` (for efficient arrays).
3. Set the desired simulation parameters at the bottom of `particle-streams.jl`.
4. Run `particle-streams.jl` to simluate in real-time. To avoid recompilation every time, launch `julia` first, then `include("particle-streams.jl")` after changing parameters to re-simulate without re-compiling.

### Basic example

https://user-images.githubusercontent.com/10370860/121208094-193e2300-c87a-11eb-9627-9c2f3b624532.mp4

### Two colliding streams above a hard wall

https://user-images.githubusercontent.com/10370860/121208105-1b07e680-c87a-11eb-838c-049b0294a35f.mp4

### Two colliding streams in a junction

https://user-images.githubusercontent.com/10370860/121208109-1c391380-c87a-11eb-829a-e3a41d1aad20.mp4
