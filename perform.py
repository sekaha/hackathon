import pstats

p = pstats.Stats("profile_results.prof")
# p.sort_stats("cumulative").print_stats(20)

# Start with draw_self() in entities.py → It’s the biggest bottleneck.
# Convert draw_self() in asteroid.py and player.py.
# Optimize draw_line() and hud() if still slow.
# Convert update() logic inside bullet.py.

total_time = p.total_tt 
frame_count = p.stats[('c:\\Users\\david\\Desktop\\Programming\\Hackathon\\run.py', 11, 'update')][0] 
fps = frame_count / total_time if total_time > 0 else 0
print(f"Estimated FPS: {fps:.2f}")