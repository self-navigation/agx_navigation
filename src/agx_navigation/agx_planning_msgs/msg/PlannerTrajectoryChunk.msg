# One streamed chunk of an offline-planned trajectory for a unicycle base.
#
# Twists are sampled uniformly at dt seconds. Within a single trajectory_id
# the interpreter plays samples in (chunk_index, sample-within-chunk) order;
# is_final marks the last chunk -- on receipt the consumer knows no more
# samples will arrive for this trajectory_id. A new trajectory_id supersedes
# any in-flight playback (e.g. on replan).
#
# The five sample arrays are parallel (equal length). The i-th sample is the
# command to apply STARTING at the i-th tick of dt; pose_x[i] / pose_y[i] /
# pose_theta[i] is the planner's predicted base pose at that tick (i.e. where
# the chassis WILL BE just before applying twist i, assuming perfect tracking
# of the previous samples). pose_theta is in radians. The final chunk for a
# trajectory may carry empty arrays -- it just signals end-of-trajectory.

std_msgs/Header header

uint64 trajectory_id    # bumped on every new plan or replan
uint32 chunk_index      # 0, 1, 2, ... within a trajectory_id
bool   is_final         # true on the last chunk for this trajectory_id

float32 dt              # [s] interval between consecutive samples

float32[] linear_x      # forward velocity v
float32[] angular_z     # yaw rate omega
float32[] pose_x        # predicted x in header.frame_id
float32[] pose_y        # predicted y in header.frame_id
float32[] pose_theta    # predicted yaw [rad]
