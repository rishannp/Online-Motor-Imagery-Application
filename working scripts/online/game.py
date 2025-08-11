import os
import pickle
import random
import pygame
import queue
import numpy as np

from config import (
    SUBJECT_DIR, SESSION_DIR,
    NUM_LEVELS, TRIALS_PER_LEVEL,
    CUE_DURATION, TRIAL_DURATION
)

# ─── GUI constants ────────────────────────────────────────────────────────
FPS = 60
PLAYER_SPEED  = 2
PLAYER_RADIUS = 30

# Colors
BG           = (31, 41, 51)
PLAYER_COLOR = (59, 130, 246)
GREY         = (128, 128, 128)
YELLOW       = (255, 255, 0)
TEXT_COLOR   = (229, 231, 235)

# Screen size
SCREEN_W, SCREEN_H = 800, 600

# Paddle geometry
PADDLE_W, PADDLE_H = 10, 100
LEFT_X            = 20
RIGHT_X           = SCREEN_W - 20 - PADDLE_W
DOT_CENTER_X      = SCREEN_W // 2
DOT_Y             = SCREEN_H // 2

def ms(x):
    return int(x * 1000)

def circle_rect_collision(cx, cy, radius, rect):
    closest_x = max(rect.left,  min(cx, rect.right))
    closest_y = max(rect.top,   min(cy, rect.bottom))
    dx = cx - closest_x
    dy = cy - closest_y
    return dx*dx + dy*dy <= radius*radius

def run_game(action_queue, adapt_queue, game_states, label_queue, raw_eeg_log, stop_event):
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Lane Runner (Press K or Esc to quit)")
    font = pygame.font.Font(None, 36)

    def spawn_list():
        s = [0] * (TRIALS_PER_LEVEL // 2) + [1] * (TRIALS_PER_LEVEL // 2)
        if TRIALS_PER_LEVEL % 2:
            s.append(random.choice([0,1]))
        random.shuffle(s)
        return s

    # ─── Session state ────────────────────────────────────────────────────
    level, trial_in = 1, 0
    hits, misses    = 0, 0
    spawns          = spawn_list()
    side            = spawns[trial_in]

    cue_ms      = ms(CUE_DURATION)
    trial_ms    = ms(TRIAL_DURATION)
    trial_start = pygame.time.get_ticks()

    DOT_X            = DOT_CENTER_X
    trial_wins       = []
    last_win         = None
    cursor_positions = []
    trials           = {}
    trial_id         = 0

    clock = pygame.time.Clock()
    run   = True
    last_cmd   = None
    last_adapt  = 0
    adapt_dur   = 0

    while run and level <= NUM_LEVELS and not stop_event.is_set():
        ts = pygame.time.get_ticks()
        screen.fill(BG)

        # ─ Adaptation indicator ─
        try:
            d = adapt_queue.get_nowait()
            last_adapt, adapt_dur = ts, d
        except queue.Empty:
            pass

        # ─ Event handling ─
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                stop_event.set()
                run = False
            elif e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_k, pygame.K_ESCAPE):
                    stop_event.set()
                    run = False

        if stop_event.is_set():
            break

        # ─ BCI command ─
        try:
            cmd = action_queue.get_nowait()
            last_cmd = cmd
        except queue.Empty:
            cmd = last_cmd

        # ─ Move the dot ─
        if cmd == 0 and DOT_X - PLAYER_RADIUS > LEFT_X + PADDLE_W:
            DOT_X -= PLAYER_SPEED
        if cmd == 1 and DOT_X + PLAYER_RADIUS < RIGHT_X:
            DOT_X += PLAYER_SPEED

        cursor_positions.append(DOT_X)

        # ─ Log EEG for adaptation ─
        if raw_eeg_log:
            w = raw_eeg_log[0]
            if id(w) != last_win:
                trial_wins.append(w.copy())
                last_win = id(w)

        # ─ Paddle coloring ─
        elapsed = ts - trial_start
        if elapsed < trial_ms:
            left_color  = YELLOW if side == 0 else GREY
            right_color = YELLOW if side == 1 else GREY
        else:
            left_color = right_color = GREY

        # ─ Draw paddles & dot ─
        left_rect  = pygame.Rect(LEFT_X, DOT_Y - PADDLE_H//2, PADDLE_W, PADDLE_H)
        right_rect = pygame.Rect(RIGHT_X, DOT_Y - PADDLE_H//2, PADDLE_W, PADDLE_H)
        pygame.draw.rect(screen, left_color,  left_rect,  border_radius=5)
        pygame.draw.rect(screen, right_color, right_rect, border_radius=5)
        pygame.draw.circle(screen, PLAYER_COLOR, (int(DOT_X), DOT_Y), PLAYER_RADIUS)

        # ─ UI text ─
        screen.blit(font.render(f"Level {level}/{NUM_LEVELS}", True, TEXT_COLOR), (10,10))
        screen.blit(font.render(f"Hits {hits}  Misses {misses}", True, TEXT_COLOR), (10,50))
        adapt_c = (0,255,0) if (ts - last_adapt) < adapt_dur else GREY
        screen.blit(font.render("Adapting", True, adapt_c), (260,10))
        screen.blit(font.render("Press K/Esc to quit", True, GREY), (10, 90))

        pygame.display.flip()
        clock.tick(FPS)

        # ─ Collision or timeout ─
        outcome = None
        target_rect     = left_rect if side == 0 else right_rect
        distractor_rect = right_rect if side == 0 else left_rect

        if circle_rect_collision(DOT_X, DOT_Y, PLAYER_RADIUS, target_rect):
            outcome, hits = 'hit', hits + 1
        elif circle_rect_collision(DOT_X, DOT_Y, PLAYER_RADIUS, distractor_rect):
            outcome, misses = 'miss', misses + 1
        elif elapsed >= trial_ms:
            outcome, misses = 'miss', misses + 1

        # ─ End-of-trial ─
        if outcome:
            label_queue.put((side, trial_wins.copy()))

            eeg_arr = np.concatenate(trial_wins, axis=0).T if trial_wins else np.empty((0,0))
            trials[trial_id] = {
                'eeg'      : eeg_arr,
                'label'    : side,
                'cursor_x' : np.array(cursor_positions),
                'hit'      : (outcome == 'hit')
            }
            trial_id += 1

            # reset
            trial_wins.clear()
            cursor_positions.clear()
            last_win = None
            DOT_X     = DOT_CENTER_X

            trial_in += 1
            if trial_in >= TRIALS_PER_LEVEL:
                trial_in = 0
                level  += 1
                spawns  = spawn_list()
            side = spawns[trial_in]
            trial_start = ts

    # ─ Save session data ─
    if trials:
        os.makedirs(SESSION_DIR, exist_ok=True)
        with open(os.path.join(SESSION_DIR, "session_data.pkl"), "wb") as f:
            pickle.dump(trials, f)

    pygame.quit()
