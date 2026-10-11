# Sourced by job runners: phone pushes for job start, a periodic heartbeat, and
# finish (or death without finishing). Built into the runner rather than a
# separate watcher so it cannot be forgotten (user, 2026-10-10).
# Server: https://ntfy.danya02.ru/agx-jobs; token ~/.config/ntfy/agx-jobs.token
# on the VM (write-only). No token -> every call is a silent no-op, so runs on
# a machine without it still work. Env: NTFY_URL, NTFY_TOKEN_FILE,
# NTFY_HEARTBEAT_H (default 1; 0 disables), NTFY_POLL_S, NTFY_STALL_MIN,
# NTFY_DISABLE=1.

_NTFY_URL=${NTFY_URL:-https://ntfy.danya02.ru/agx-jobs}
_NTFY_TOKEN_FILE=${NTFY_TOKEN_FILE:-$HOME/.config/ntfy/agx-jobs.token}
_NTFY_HB_PID=""
_NTFY_FINISHED=0

ntfy_push() {  # <title> <priority> <tags> <message>; never fails the caller
    [ "${NTFY_DISABLE:-0}" = 1 ] && return 0
    [ -r "$_NTFY_TOKEN_FILE" ] || return 0
    local i
    for i in 1 2 3; do
        curl -fsS --max-time 20 -o /dev/null \
            -H "Authorization: Bearer $(<"$_NTFY_TOKEN_FILE")" -H "Title: $1" \
            -H "Priority: $2" -H "Tags: $3" --data-binary "$4" "$_NTFY_URL" && return 0
        sleep 10
    done
    echo "[ntfy] push failed: $1" >&2
    return 0
}

# ntfy_job_begin <job name> <progress-cmd> [<count-cmd> <total>]: push
# "started" and fork a watcher that wakes every NTFY_POLL_S seconds (default
# 300). With a count-cmd (prints rows done so far) it pushes on CONDITIONS:
#   - each 25% milestone crossed;
#   - STALLED when the count has not moved for NTFY_STALL_MIN minutes (default
#     20; a 12-worker job adds a row every ~10 s), and "resumed" when it moves;
# plus the plain heartbeat every NTFY_HEARTBEAT_H hours (default 1; 0
# disables), so silence for longer than that means the watcher itself is gone.
ntfy_job_begin() {
    _NTFY_JOB=$1; _NTFY_T0=$(date +%s)
    ntfy_push "agx: $_NTFY_JOB started" low arrow_forward "$(eval "$2")"
    local h=${NTFY_HEARTBEAT_H:-1} poll=${NTFY_POLL_S:-300} stall=${NTFY_STALL_MIN:-20}
    ( trap - EXIT
      local hb_s next_hb last_n=-1 last_move milestone=0 stalled=0 n m now
      hb_s=$(awk "BEGIN{print int($h*3600)}"); next_hb=$(( _NTFY_T0 + hb_s ))
      local stall_s; stall_s=$(awk "BEGIN{print int($stall*60)}")
      last_move=$(date +%s)
      while sleep "$poll"; do
          now=$(date +%s)
          local mins=$(( (now - _NTFY_T0) / 60 ))
          if [ -n "${3:-}" ] && [ "${4:-0}" -gt 0 ]; then
              n=$(eval "$3" 2>/dev/null); n=${n:-0}
              if [ "$n" != "$last_n" ]; then
                  last_n=$n; last_move=$now
                  if [ "$stalled" = 1 ]; then
                      stalled=0
                      ntfy_push "agx: $_NTFY_JOB resumed ($mins min)" default arrow_forward "$(eval "$2")"
                  fi
              elif [ "$stalled" = 0 ] && [ $(( now - last_move )) -ge "$stall_s" ]; then
                  stalled=1
                  ntfy_push "agx: $_NTFY_JOB STALLED: no new rows for $(( (now - last_move) / 60 )) min" \
                      high warning "$(eval "$2")"
              fi
              m=$(( 4 * n / $4 ))
              if [ "$m" -gt "$milestone" ] && [ "$m" -lt 4 ]; then
                  milestone=$m
                  ntfy_push "agx: $_NTFY_JOB past $(( 25 * m ))% ($mins min)" low hourglass "$(eval "$2")"
                  next_hb=$(( now + hb_s ))   # a milestone is a heartbeat
              fi
          fi
          if [ "$hb_s" -gt 0 ] && [ "$now" -ge "$next_hb" ]; then
              next_hb=$(( now + hb_s ))
              ntfy_push "agx: $_NTFY_JOB running $mins min" min hourglass "$(eval "$2")"
          fi
      done ) &
    _NTFY_HB_PID=$!
}

_ntfy_stop_heartbeat() {
    [ -n "$_NTFY_HB_PID" ] && kill "$_NTFY_HB_PID" 2>/dev/null
    _NTFY_HB_PID=""
}

# ntfy_job_finished <message> <n_failed>: the normal end. Stops the heartbeat
# itself: runners drop their EXIT trap after a clean finish, so leaving this to
# ntfy_job_exit kept heartbeats pushing "running ... 100%" for days (2026-10-11).
ntfy_job_finished() {
    _NTFY_FINISHED=1
    _ntfy_stop_heartbeat
    local mins=$(( ($(date +%s) - ${_NTFY_T0:-$(date +%s)}) / 60 ))
    if [ "${2:-0}" -gt 0 ]; then
        ntfy_push "agx: $_NTFY_JOB finished ($mins min), $2 failed" high warning "$1"
    else
        ntfy_push "agx: $_NTFY_JOB finished ($mins min)" default white_check_mark "$1"
    fi
}

# Call from the runner's EXIT trap: stops the heartbeat; if the job never
# reached ntfy_job_finished, says so loudly.
ntfy_job_exit() {
    _ntfy_stop_heartbeat
    [ "$_NTFY_FINISHED" = 1 ] || [ -z "${_NTFY_JOB:-}" ] || \
        ntfy_push "agx: $_NTFY_JOB DIED" urgent rotating_light \
            "Runner exited without finishing (rc=${1:-?}). Check its runner.log."
}
