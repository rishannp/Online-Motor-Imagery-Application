# game.py

import os, pickle, random, pygame, queue, numpy as np
from config import (
    SUBJECT_DIR, SESSION_DIR,
    NUM_LEVELS, TRIALS_PER_LEVEL,
    CUE_DURATION, TRIAL_DURATION,
    SAMPLING_RATE
)

# Visuals
FPS = 60
PLAYER_SPEED, PLAYER_RADIUS = 2, 30
BG, PLAYER_COLOR = (31,41,51), (59,130,246)
GREY, YELLOW, TEXT_COLOR = (128,128,128), (255,255,0), (229,231,235)

# Screen & geometry
SCREEN_W, SCREEN_H = 800, 600
PADDLE_W, PADDLE_H = 10, 100
LEFT_X, RIGHT_X    = 20, SCREEN_W - 20 - PADDLE_W
DOT_CENTER_X, DOT_Y = SCREEN_W // 2, SCREEN_H // 2

def ms(x): return int(x * 1000)

def circle_rect_collision(cx, cy, r, rect):
    closest_x = max(rect.left, min(cx, rect.right))
    closest_y = max(rect.top,  min(cy, rect.bottom))
    dx, dy = cx - closest_x, cy - closest_y
    return dx*dx + dy*dy <= r*r

def _drain_queue(q):
    """Non-blocking drain of all pending items in a Queue."""
    drained = 0
    while True:
        try:
            q.get_nowait()
            drained += 1
        except queue.Empty:
            return drained

def run_game(action_q, adapt_q, game_states, label_q, raw_eeg_q, eeg_chunk_q):
    """
    NEW: `eeg_chunk_q` streams every incoming EEG sample as a [1, n_ch] array.
    We accumulate those per-trial to produce ONE contiguous buffer per trial.

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
        s = [0]*(TRIALS_PER_LEVEL//2) + [1]*(TRIALS_PER_LEVEL//2)
        if TRIALS_PER_LEVEL % 2: s.append(random.choice([0,1]))
        random.shuffle(s); return s

    level, ti, hits, misses = 1, 0, 0, 0
    spawns = spawn_list(); side = spawns[ti]
    cue_ms, trial_ms = ms(CUE_DURATION), ms(TRIAL_DURATION)
    trial_start = pygame.time.get_ticks()

    DOT_X = DOT_CENTER_X
    # For adaptation (if used elsewhere): keep the last window list
    trial_wins, last_win = [], None
    # Continuous EEG collector: list of [chunk_len, n_ch] arrays
    trial_eeg_chunks = []
    cursor_positions = []
    trials, tid = {}, 0

    # Before starting, make sure no stale chunks are in the queue
    _drain_queue(eeg_chunk_q)

    clock = pygame.time.Clock()
    run = True; last_cmd = None; last_adapt = 0; adapt_dur = 0

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

        # Get latest BCI command (0/1)
        try:
            cmd = action_q.get_nowait(); last_cmd = cmd
        except queue.Empty:
            cmd = last_cmd

        # Move the dot with bounds
        if cmd == 0 and DOT_X - PLAYER_RADIUS > LEFT_X + PADDLE_W: DOT_X -= PLAYER_SPEED
        if cmd == 1 and DOT_X + PLAYER_RADIUS < RIGHT_X:           DOT_X += PLAYER_SPEED

        cursor_positions.append(DOT_X)

        # For adaptation (window snapshots)
        if raw_eeg_q:
            w = raw_eeg_q[0]
            if id(w) != last_win:
                trial_wins.append(w.copy()); last_win = id(w)

        # CONTINUOUS EEG: drain all new chunks for this frame
        while True:
            try:
                chunk = eeg_chunk_q.get_nowait()   # [k, n_ch], usually [1, n_ch]
                trial_eeg_chunks.append(chunk)
            except queue.Empty:
                break

        # Coloring & draw
        elapsed = ts - trial_start
        if elapsed < trial_ms:
            lc, rc = (YELLOW if side == 0 else GREY), (YELLOW if side == 1 else GREY)
        else:
            lc = rc = GREY

        left_rect  = pygame.Rect(LEFT_X,  DOT_Y - PADDLE_H//2, PADDLE_W, PADDLE_H)
        right_rect = pygame.Rect(RIGHT_X, DOT_Y - PADDLE_H//2, PADDLE_W, PADDLE_H)
        pygame.draw.rect(screen, lc, left_rect,  border_radius=5)
        pygame.draw.rect(screen, rc, right_rect, border_radius=5)
        pygame.draw.circle(screen, PLAYER_COLOR, (int(DOT_X), DOT_Y), PLAYER_RADIUS)

        screen.blit(font.render(f"Level {level}/{NUM_LEVELS}", True, TEXT_COLOR), (10,10))
        screen.blit(font.render(f"Hits {hits}  Misses {misses}", True, TEXT_COLOR), (10,50))
        ac = (0,255,0) if (ts - last_adapt) < adapt_dur else GREY
        screen.blit(font.render("Adapting", True, ac), (260,10))

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
            # (optional) adaptation windows
            label_q.put((side, trial_wins.copy()))

            # Build ONE continuous EEG array for the trial
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

            # Reset per-trial collectors
            trial_wins.clear()
            last_win = None
            trial_eeg_chunks.clear()
            cursor_positions.clear()
            DOT_X = DOT_CENTER_X

            # Start next trial — drain any stale chunks that arrived between trials
            _drain_queue(eeg_chunk_q)

            ti += 1
            if ti >= TRIALS_PER_LEVEL:
                ti = 0; level += 1; spawns = spawn_list()
            side = spawns[ti]
            trial_start = ts

    # Save all trials
    os.makedirs(SESSION_DIR, exist_ok=True)
    with open(os.path.join(SESSION_DIR, "session_data.pkl"), "wb") as f:
        pickle.dump(trials, f)

    pygame.quit()
