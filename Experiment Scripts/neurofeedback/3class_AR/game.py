import os, pickle, random, pygame, queue, numpy as np
from config import (
    SUBJECT_DIR, SESSION_DIR,
    NUM_LEVELS, TRIALS_PER_LEVEL,
    CUE_DURATION, TRIAL_DURATION,
    SAMPLING_RATE,
    REST_BASELINE_SEC,
    INTER_TRIAL_PAUSE,     # inter-trial pause (s)
    INTER_LEVEL_PAUSE      # inter-level pause (s)
)

# -------------------------------
# Visual / gameplay parameters
# -------------------------------
FPS = 60
PLAYER_SPEED, PLAYER_RADIUS = 2, 30
BG, PLAYER_COLOR = (31, 41, 51), (59, 130, 246)
GREY, YELLOW, TEXT_COLOR = (128, 128, 128), (255, 255, 0), (229, 231, 235)

# Screen & geometry
SCREEN_W, SCREEN_H = 800, 600
PADDLE_W, PADDLE_H = 10, 100
LEFT_X, RIGHT_X    = 20, SCREEN_W - 20 - PADDLE_W
DOT_CENTER_X, DOT_Y = SCREEN_W // 2, SCREEN_H // 2

# I use a subtle blink on the preview paddle so the next target is obvious but not noisy
PREVIEW_BLINK_MS = 400

def ms(x): 
    # I convert seconds to milliseconds for pygame timing
    return int(x * 1000)

def circle_rect_collision(cx, cy, r, rect):
    # I do circle-rect collision by clamping to the rect, then checking distance to the circle center
    closest_x = max(rect.left, min(cx, rect.right))
    closest_y = max(rect.top,  min(cy, rect.bottom))
    dx, dy = cx - closest_x, cy - closest_y
    return dx*dx + dy*dy <= r*r

def _drain_queue(q):
    """I non-blockingly drain all pending items in a Queue so nothing contaminates between states."""
    drained = 0
    while True:
        try:
            q.get_nowait()
            drained += 1
        except queue.Empty:
            return drained

def _draw_paddles(screen, left_color, right_color):
    """I centralize paddle drawing so I don't duplicate rect math."""
    left_rect  = pygame.Rect(LEFT_X,  DOT_Y - PADDLE_H//2, PADDLE_W, PADDLE_H)
    right_rect = pygame.Rect(RIGHT_X, DOT_Y - PADDLE_H//2, PADDLE_W, PADDLE_H)
    pygame.draw.rect(screen, left_color,  left_rect,  border_radius=5)
    pygame.draw.rect(screen, right_color, right_rect, border_radius=5)
    return left_rect, right_rect

def run_game(action_q, adapt_q, game_states, label_q, raw_eeg_q, eeg_chunk_q):
    """
    `eeg_chunk_q` streams every incoming EEG sample as a [1, n_ch] array.
    I accumulate those per-trial to produce ONE contiguous buffer per trial.

    Saved per trial:
      trials[tid] = {
        'eeg':      np.ndarray, shape [n_channels, n_samples], continuous,
        'fs':       int, sampling rate (Hz),
        'label':    int (0=Left, 1=Right),
        'cursor_x': np.ndarray [n_frames], dot x per rendered frame,
        'hit':      bool
      }
    """
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Lane Runner (Neurofeedback)")
    font = pygame.font.Font(None, 36)

    def spawn_list():
        # I construct a balanced list of L/R targets and shuffle
        s = [0]*(TRIALS_PER_LEVEL//2) + [1]*(TRIALS_PER_LEVEL//2)
        if TRIALS_PER_LEVEL % 2:
            s.append(random.choice([0, 1]))
        random.shuffle(s)
        return s

    # --- basic timings ---
    cue_ms   = ms(CUE_DURATION)
    trial_ms = ms(TRIAL_DURATION)
    iti_ms   = ms(float(INTER_TRIAL_PAUSE))   # inter-trial pause (ms)
    ilp_ms   = ms(float(INTER_LEVEL_PAUSE))   # inter-level pause (ms)
    baseline_ms = ms(float(REST_BASELINE_SEC))

    # --- state machine for baseline / trial / inter-trial / inter-level ---
    STATE_BASELINE = 0
    STATE_TRIAL    = 1
    STATE_ITI      = 2
    STATE_ILP      = 3   # inter-level pause
    state = STATE_TRIAL

    level, ti, hits, misses = 1, 0, 0, 0
    spawns = spawn_list()
    side = spawns[ti]  # 0 = LEFT, 1 = RIGHT
    trial_start = pygame.time.get_ticks()
    baseline_start = None
    iti_start = None
    ilp_start = None

    DOT_X = DOT_CENTER_X
    # For adaptation (if used elsewhere): I keep the last window list
    trial_wins, last_win = [], None
    # Continuous EEG collector: list of [chunk_len, n_ch] arrays
    trial_eeg_chunks = []
    cursor_positions = []
    trials, tid = {}, 0

    # Before starting, I make sure no stale chunks are in the queue
    _drain_queue(eeg_chunk_q)

    clock = pygame.time.Clock()
    run = True
    last_cmd = None
    last_adapt = 0
    adapt_dur = 0

    # I start every level with an explicit baseline rest block so the decoder can
    # estimate no-control stats defensibly.
    state = STATE_BASELINE
    baseline_start = pygame.time.get_ticks()
    if game_states:
        game_states.put('BASELINE_START')

    while run and level <= NUM_LEVELS:
        ts = pygame.time.get_ticks()
        screen.fill(BG)

        # (optional) adaptation timing indicator
        try:
            d = adapt_q.get_nowait()
            last_adapt, adapt_dur = ts, d
        except queue.Empty:
            pass

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                run = False

        # ------------- STATE: BASELINE (rest) -------------
        if state == STATE_BASELINE:
            # I ignore commands, discard EEG, and show a clear baseline screen
            _drain_queue(eeg_chunk_q)

            pygame.draw.circle(screen, PLAYER_COLOR, (int(DOT_X), DOT_Y), PLAYER_RADIUS)

            screen.blit(font.render(f"Level {level}/{NUM_LEVELS}", True, TEXT_COLOR), (10, 10))
            screen.blit(font.render("Baseline Rest", True, TEXT_COLOR), (DOT_CENTER_X - 110, 20))

            remain_ms = max(0, ms(float(REST_BASELINE_SEC)) - (ts - baseline_start))
            screen.blit(font.render(f"{remain_ms/1000:.1f}s", True, TEXT_COLOR),
                        (DOT_CENTER_X - 20, 60))

            pygame.display.flip()
            clock.tick(FPS)

            if ts - baseline_start >= ms(float(REST_BASELINE_SEC)):
                if game_states:
                    game_states.put('BASELINE_END')
                trial_start = ts
                state = STATE_TRIAL
                _drain_queue(eeg_chunk_q)
                continue

        # ------------- STATE: TRIAL -------------
        elif state == STATE_TRIAL:
            # I get latest BCI command (0/1) and hold last if queue empty
            try:
                cmd = action_q.get_nowait()
                last_cmd = cmd
            except queue.Empty:
                cmd = last_cmd

            # I move the dot with bounds (only during trial)
            if cmd == 0 and DOT_X - PLAYER_RADIUS > LEFT_X + PADDLE_W:
                DOT_X -= PLAYER_SPEED
            if cmd == 1 and DOT_X + PLAYER_RADIUS < RIGHT_X:
                DOT_X += PLAYER_SPEED
            # cmd == -1 (no-control) → I don't move the cursor

            cursor_positions.append(DOT_X)

            # For adaptation (window snapshots) – only during trial
            if raw_eeg_q:
                w = raw_eeg_q[0]
                if id(w) != last_win:
                    trial_wins.append(w.copy())
                    last_win = id(w)

            # CONTINUOUS EEG: I drain all new chunks for this frame (trial only)
            while True:
                try:
                    chunk = eeg_chunk_q.get_nowait()   # [k, n_ch], usually [1, n_ch]
                    trial_eeg_chunks.append(chunk)
                except queue.Empty:
                    break

            # Coloring & draw
            elapsed = ts - trial_start
            if elapsed < trial_ms:
                lc = YELLOW if side == 0 else GREY
                rc = YELLOW if side == 1 else GREY
            else:
                lc = rc = GREY

            left_rect, right_rect = _draw_paddles(screen, lc, rc)
            pygame.draw.circle(screen, PLAYER_COLOR, (int(DOT_X), DOT_Y), PLAYER_RADIUS)

            screen.blit(font.render(f"Level {level}/{NUM_LEVELS}", True, TEXT_COLOR), (10, 10))
            screen.blit(font.render(f"Hits {hits}  Misses {misses}", True, TEXT_COLOR), (10, 50))
            ac = (0, 255, 0) if (ts - last_adapt) < adapt_dur else GREY
            screen.blit(font.render("Adapting", True, ac), (260, 10))

            pygame.display.flip()
            clock.tick(FPS)

            # Outcome check
            outcome = None
            tgt  = left_rect  if side == 0 else right_rect
            dist = right_rect if side == 0 else left_rect
            if circle_rect_collision(DOT_X, DOT_Y, PLAYER_RADIUS, tgt):
                outcome, hits = 'hit', hits + 1
            elif circle_rect_collision(DOT_X, DOT_Y, PLAYER_RADIUS, dist):
                outcome, misses = 'miss', misses + 1
            elif elapsed >= trial_ms:
                outcome, misses = 'miss', misses + 1

            if outcome:
                # (optional) I snapshot adaptation windows per trial
                label_q.put((side, trial_wins.copy()))

                # I build ONE continuous EEG array for the trial
                if len(trial_eeg_chunks) > 0:
                    cont = np.vstack(trial_eeg_chunks)          # [n_samples, n_ch]
                    eeg_arr = cont.T.astype(np.float32)         # [n_ch, n_samples]
                else:
                    eeg_arr = np.empty((0, 0), dtype=np.float32)

                trials[tid] = {
                    'eeg'      : eeg_arr,
                    'fs'       : int(SAMPLING_RATE),
                    'label'    : side,                 # 0=Left, 1=Right (target side)
                    'cursor_x' : np.array(cursor_positions, dtype=np.float32),
                    'hit'      : (outcome == 'hit')
                }
                tid += 1

                # I prepare for rest (reset collectors, freeze scene)
                trial_wins.clear()
                last_win = None
                trial_eeg_chunks.clear()
                cursor_positions.clear()
                DOT_X = DOT_CENTER_X

                # I drain any EEG that arrives *between* trials/levels so we don't contaminate
                _drain_queue(eeg_chunk_q)

                # I advance trial index / possibly level
                ti += 1
                if ti >= TRIALS_PER_LEVEL:
                    # Finished this level
                    ti = 0
                    level += 1
                    if level > NUM_LEVELS:
                        # all done — leave loop
                        break
                    # I prepare next level's spawn sequence
                    spawns = spawn_list()
                    side = spawns[ti]  # first target of next level (will start after ILP)
                    # Switch to inter-level pause
                    state = STATE_ILP
                    ilp_start = ts
                    continue
                else:
                    # Still within same level: set next target and go to ITI
                    side = spawns[ti]
                    state = STATE_ITI
                    iti_start = ts
                    continue

        # ------------- STATE: ITI (inter-trial pause) -------------
        elif state == STATE_ITI:
            # I ignore commands, discard EEG, and show a clear rest screen
            _drain_queue(eeg_chunk_q)

            # I blink the preview paddle so it's obvious but not annoying
            blink_on = ((ts // PREVIEW_BLINK_MS) % 2) == 0
            # I highlight the *next* target (side already set to next trial above)
            lc = YELLOW if (side == 0 and blink_on) else GREY
            rc = YELLOW if (side == 1 and blink_on) else GREY

            left_rect, right_rect = _draw_paddles(screen, lc, rc)

            # I leave the dot parked in the middle during rest
            pygame.draw.circle(screen, PLAYER_COLOR, (int(DOT_X), DOT_Y), PLAYER_RADIUS)

            # HUD
            screen.blit(font.render(f"Level {level}/{NUM_LEVELS}", True, TEXT_COLOR), (10, 10))
            screen.blit(font.render(f"Hits {hits}  Misses {misses}", True, TEXT_COLOR), (10, 50))
            screen.blit(font.render("Rest", True, TEXT_COLOR), (DOT_CENTER_X - 30, 20))

            # I show explicit next target + countdown so participants can anticipate
            remain_ms = max(0, iti_ms - (ts - iti_start))
            next_txt  = "LEFT" if side == 0 else "RIGHT"
            screen.blit(font.render(f"Next target: {next_txt}", True, TEXT_COLOR),
                        (DOT_CENTER_X - 110, 60))
            screen.blit(font.render(f"{remain_ms/1000:.1f}s", True, TEXT_COLOR),
                        (DOT_CENTER_X - 20, 90))

            pygame.display.flip()
            clock.tick(FPS)

            if ts - iti_start >= iti_ms:
                trial_start = ts
                state = STATE_TRIAL
                _drain_queue(eeg_chunk_q)
                continue

        # ------------- STATE: ILP (inter-level pause) -------------
        elif state == STATE_ILP:
            # I ignore commands, discard EEG, and indicate level complete + next target
            _drain_queue(eeg_chunk_q)

            blink_on = ((ts // PREVIEW_BLINK_MS) % 2) == 0
            # Note: 'side' is already set to the first target of the next level
            lc = YELLOW if (side == 0 and blink_on) else GREY
            rc = YELLOW if (side == 1 and blink_on) else GREY

            left_rect, right_rect = _draw_paddles(screen, lc, rc)
            pygame.draw.circle(screen, PLAYER_COLOR, (int(DOT_X), DOT_Y), PLAYER_RADIUS)

            screen.blit(font.render(f"Level {level-1} complete", True, TEXT_COLOR), (10, 10))
            screen.blit(font.render(f"Next: Level {level}/{NUM_LEVELS}", True, TEXT_COLOR), (10, 50))
            screen.blit(font.render("Level Complete — Rest", True, TEXT_COLOR),
                        (DOT_CENTER_X - 160, 20))

            # I show the upcoming side + countdown
            remain_ms = max(0, ilp_ms - (ts - ilp_start))
            next_txt  = "LEFT" if side == 0 else "RIGHT"
            screen.blit(font.render(f"Next target: {next_txt}", True, TEXT_COLOR),
                        (DOT_CENTER_X - 110, 60))
            screen.blit(font.render(f"{remain_ms/1000:.1f}s", True, TEXT_COLOR),
                        (DOT_CENTER_X - 20, 90))

            pygame.display.flip()
            clock.tick(FPS)

            if ts - ilp_start >= ilp_ms:
                # I start the next level with a fresh baseline rest block
                state = STATE_BASELINE
                baseline_start = ts
                if game_states:
                    game_states.put('BASELINE_START')
                _drain_queue(eeg_chunk_q)
                # 'side' stays as the first target for the new level
                continue

    # I save all trials at the end of the session
    os.makedirs(SESSION_DIR, exist_ok=True)
    with open(os.path.join(SESSION_DIR, "session_data.pkl"), "wb") as f:
        pickle.dump(trials, f)

    pygame.quit()
