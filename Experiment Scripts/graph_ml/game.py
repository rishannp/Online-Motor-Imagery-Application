# game.py
import os, pickle, random, pygame, queue, numpy as np
from config import (
    CURRENT_SESSION_DIR,
    NUM_LEVELS, TRIALS_PER_LEVEL,
    CUE_DURATION, TRIAL_DURATION,
    SAMPLING_RATE,
    INTER_TRIAL_PAUSE,
    INTER_LEVEL_PAUSE
)

FPS = 60
PLAYER_SPEED, PLAYER_RADIUS = 2, 30
BG, PLAYER_COLOR = (31, 41, 51), (59, 130, 246)
GREY, YELLOW, TEXT_COLOR = (128, 128, 128), (255, 255, 0), (229, 231, 235)

SCREEN_W, SCREEN_H = 800, 600
PADDLE_W, PADDLE_H = 10, 100
LEFT_X, RIGHT_X    = 20, SCREEN_W - 20 - PADDLE_W
DOT_CENTER_X, DOT_Y = SCREEN_W // 2, SCREEN_H // 2

PREVIEW_BLINK_MS = 400

def ms(x):
    return int(x * 1000)

def circle_rect_collision(cx, cy, r, rect):
    closest_x = max(rect.left, min(cx, rect.right))
    closest_y = max(rect.top,  min(cy, rect.bottom))
    dx, dy = cx - closest_x, cy - closest_y
    return dx*dx + dy*dy <= r*r

def _drain_queue(q):
    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            return

def _draw_paddles(screen, left_color, right_color):
    left_rect  = pygame.Rect(LEFT_X,  DOT_Y - PADDLE_H//2, PADDLE_W, PADDLE_H)
    right_rect = pygame.Rect(RIGHT_X, DOT_Y - PADDLE_H//2, PADDLE_W, PADDLE_H)
    pygame.draw.rect(screen, left_color,  left_rect,  border_radius=5)
    pygame.draw.rect(screen, right_color, right_rect, border_radius=5)
    return left_rect, right_rect

def _drain_latest_cmd(action_q):
    newest = None
    while True:
        try:
            newest = action_q.get_nowait()
        except queue.Empty:
            break
    return newest

def run_game(action_q, adapt_q, game_states, label_q, raw_eeg_q, eeg_chunk_q):
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Lane Runner (Graph+ML)")
    font = pygame.font.Font(None, 36)

    def spawn_list():
        s = [0]*(TRIALS_PER_LEVEL//2) + [1]*(TRIALS_PER_LEVEL//2)
        if TRIALS_PER_LEVEL % 2:
            s.append(random.choice([0, 1]))
        random.shuffle(s)
        return s

    cue_ms   = ms(CUE_DURATION)
    trial_ms = ms(TRIAL_DURATION)
    iti_ms   = ms(float(INTER_TRIAL_PAUSE))
    ilp_ms   = ms(float(INTER_LEVEL_PAUSE))

    STATE_TRIAL = 0
    STATE_ITI   = 1
    STATE_ILP   = 2
    state = STATE_TRIAL

    level, ti, hits, misses = 1, 0, 0, 0
    spawns = spawn_list()
    side = spawns[ti]
    trial_start = pygame.time.get_ticks()
    iti_start = None
    ilp_start = None

    DOT_X = DOT_CENTER_X
    trial_wins, last_win = [], None
    trial_eeg_chunks = []
    cursor_positions = []
    trials, tid = {}, 0

    _drain_queue(eeg_chunk_q)

    clock = pygame.time.Clock()
    run = True

    baseline_active = True
    last_cmd = None

    last_adapt = 0
    adapt_dur = 0

    while run and level <= NUM_LEVELS:
        ts = pygame.time.get_ticks()
        screen.fill(BG)

        try:
            d = adapt_q.get_nowait()
            last_adapt, adapt_dur = ts, d
        except queue.Empty:
            pass

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                run = False

        if state == STATE_TRIAL:
            new_cmd = _drain_latest_cmd(action_q)

            if new_cmd in (0, 1):
                last_cmd = new_cmd
                baseline_active = False
            elif baseline_active:
                last_cmd = None

            baseline_mode = baseline_active

            if baseline_mode:
                DOT_X = DOT_CENTER_X
            else:
                if last_cmd == 0 and DOT_X - PLAYER_RADIUS > LEFT_X + PADDLE_W:
                    DOT_X -= PLAYER_SPEED
                elif last_cmd == 1 and DOT_X + PLAYER_RADIUS < RIGHT_X:
                    DOT_X += PLAYER_SPEED

            cursor_positions.append(DOT_X)

            if raw_eeg_q:
                w = raw_eeg_q[0]
                if id(w) != last_win:
                    trial_wins.append(w.copy())
                    last_win = id(w)

            while True:
                try:
                    chunk = eeg_chunk_q.get_nowait()
                    trial_eeg_chunks.append(chunk)
                except queue.Empty:
                    break

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

            if baseline_mode:
                screen.blit(font.render("Baseline (Rest)", True, TEXT_COLOR),
                            (DOT_CENTER_X - 105, 20))

            pygame.display.flip()
            clock.tick(FPS)

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
                label_q.put((side, trial_wins.copy()))

                if len(trial_eeg_chunks) > 0:
                    cont = np.vstack(trial_eeg_chunks)
                    eeg_arr = cont.T.astype(np.float32)
                else:
                    eeg_arr = np.empty((0, 0), dtype=np.float32)

                trials[tid] = {
                    'eeg'      : eeg_arr,
                    'fs'       : int(SAMPLING_RATE),
                    'label'    : side,
                    'cursor_x' : np.array(cursor_positions, dtype=np.float32),
                    'hit'      : (outcome == 'hit')
                }
                tid += 1

                trial_wins.clear()
                last_win = None
                trial_eeg_chunks.clear()
                cursor_positions.clear()
                DOT_X = DOT_CENTER_X

                _drain_queue(eeg_chunk_q)

                ti += 1
                if ti >= TRIALS_PER_LEVEL:
                    ti = 0
                    level += 1
                    if level > NUM_LEVELS:
                        break
                    spawns = spawn_list()
                    side = spawns[ti]
                    state = STATE_ILP
                    ilp_start = ts
                    continue
                else:
                    side = spawns[ti]
                    state = STATE_ITI
                    iti_start = ts
                    continue

        elif state == STATE_ITI:
            _drain_queue(eeg_chunk_q)

            blink_on = ((ts // PREVIEW_BLINK_MS) % 2) == 0
            lc = YELLOW if (side == 0 and blink_on) else GREY
            rc = YELLOW if (side == 1 and blink_on) else GREY

            _draw_paddles(screen, lc, rc)
            pygame.draw.circle(screen, PLAYER_COLOR, (int(DOT_X), DOT_Y), PLAYER_RADIUS)

            screen.blit(font.render(f"Level {level}/{NUM_LEVELS}", True, TEXT_COLOR), (10, 10))
            screen.blit(font.render(f"Hits {hits}  Misses {misses}", True, TEXT_COLOR), (10, 50))
            screen.blit(font.render("Rest", True, TEXT_COLOR), (DOT_CENTER_X - 30, 20))

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

        elif state == STATE_ILP:
            _drain_queue(eeg_chunk_q)

            blink_on = ((ts // PREVIEW_BLINK_MS) % 2) == 0
            lc = YELLOW if (side == 0 and blink_on) else GREY
            rc = YELLOW if (side == 1 and blink_on) else GREY

            _draw_paddles(screen, lc, rc)
            pygame.draw.circle(screen, PLAYER_COLOR, (int(DOT_X), DOT_Y), PLAYER_RADIUS)

            screen.blit(font.render(f"Level {level-1} complete", True, TEXT_COLOR), (10, 10))
            screen.blit(font.render(f"Next: Level {level}/{NUM_LEVELS}", True, TEXT_COLOR), (10, 50))
            screen.blit(font.render("Level Complete — Rest", True, TEXT_COLOR),
                        (DOT_CENTER_X - 160, 20))

            remain_ms = max(0, ilp_ms - (ts - ilp_start))
            next_txt  = "LEFT" if side == 0 else "RIGHT"
            screen.blit(font.render(f"Next target: {next_txt}", True, TEXT_COLOR),
                        (DOT_CENTER_X - 110, 60))
            screen.blit(font.render(f"{remain_ms/1000:.1f}s", True, TEXT_COLOR),
                        (DOT_CENTER_X - 20, 90))

            pygame.display.flip()
            clock.tick(FPS)

            if ts - ilp_start >= ilp_ms:
                trial_start = ts
                state = STATE_TRIAL
                _drain_queue(eeg_chunk_q)
                continue

    os.makedirs(CURRENT_SESSION_DIR, exist_ok=True)
    with open(os.path.join(CURRENT_SESSION_DIR, "session_data.pkl"), "wb") as f:
        pickle.dump(trials, f)

    pygame.quit()
