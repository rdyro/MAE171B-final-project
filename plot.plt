set terminal pngcairo size 1000,600 font 'Arial,25'
set output 'graph/sys_id.png'

set xlabel 'Time, (s)'
set ylabel 'Angular Velocity, {/Symbol w} (rad/s)'
plot 'data/sys_id.txt' u 1:2 w l lw 3 title 'Raw', \
     'data/sys_id.txt' u 1:3 w l lw 3 title 'Filtered', \
     'data/sys_id.txt' u 1:4 w l lw 3 title 'Fit'
