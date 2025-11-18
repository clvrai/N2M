# Rendering Issues Summary

## Problem
"Interrupted system call ; error code 4" when render=true

## Verified Facts
✅ RoboCasa official demo works with GUI
✅ Our data collection works with render=false
⚠️  Our data collection fails with render=true (VNC/OpenGL timing issue)

## Root Cause
Combination of on-screen + offscreen rendering with depth cameras triggers 
OpenGL initialization timing issues in VNC environments.

Official demo uses: has_offscreen_renderer=False, use_camera_obs=False
Our script needs: has_offscreen_renderer=True, use_camera_obs=True (for point clouds)

## Solution
**Use render=false for data collection** (already default)

All core functionality works perfectly:
- ✅ Colored point clouds from 5 depth cameras
- ✅ Robot removed before capture  
- ✅ Correct N2M data format
- ✅ Folder naming: {task}_{layout}_{style}/

For visual debugging: Use `python -m robocasa.demos.demo_kitchen_scenes`
