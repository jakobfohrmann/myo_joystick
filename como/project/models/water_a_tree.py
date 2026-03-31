"""
Water a Tree – Mini-Spiel mit Overlay-Visualisierung

Rasen als Untergrund, Joystick-Position = Wassereimer, Targets = Setzlinge.
Trifft der Eimer den Setzling (in Range), wird er zum Baum → Reset, neuer Setzling.
Scoreboard: Anzahl geretteter Bäume.

Funktioniert wie das Overlay in create_video.py (gleiches Koordinatensystem [-1, 1]).
Steuerung: Tastatur (WASD) oder optional Joystick-Position aus Environment.

Mit --model und --record: ThumbReach-Env + Policy spielen 20 Runden und Aufnahme als Video.
"""
import argparse
import os
import sys
import numpy as np
import cv2


# ----- Konfiguration -----
WINDOW_W = 640
WINDOW_H = 480
FPS = 30
# Spielfeld = zentraler Bereich (wie Overlay), Rand für Scoreboard
PLAY_SIZE = 400
PLAY_MARGIN = 20
TARGET_RADIUS = 0.3    # in normierten Koordinaten [-1,1]
# Gleiche 20 Target-Positionen wie in callbacks.CurriculumCallback.TARGETS_20 (nur diese für Setzlinge)
TARGETS_20 = [
    np.array([0.82, 0.71]), np.array([-0.91, 0.58]), np.array([0.14, -0.87]), np.array([-0.76, -0.43]),
    np.array([0.61, 0.94]), np.array([-0.38, -0.72]), np.array([0.93, -0.29]), np.array([-0.55, 0.81]),
    np.array([0.27, 0.19]), np.array([-0.84, -0.16]), np.array([0.68, -0.63]), np.array([-0.22, 0.47]),
    np.array([0.45, -0.51]), np.array([-0.67, 0.33]), np.array([0.08, 0.78]), np.array([-0.96, -0.89]),
    np.array([0.53, 0.41]), np.array([-0.49, -0.56]), np.array([0.79, -0.12]), np.array([-0.11, 0.65]),
]
GRASS_COLOR = (34, 139, 34)      # BGR: ForestGreen
GRASS_DARK = (0, 100, 0)         # BGR: dunkler
CAN_COLOR = (255, 165, 0)        # BGR: blau-orange (Eimer)
CAN_BORDER = (180, 120, 0)
SAPLING_STEM = (42, 42, 128)     # BGR: braun
SAPLING_TOP = (0, 128, 0)        # BGR: grün
TREE_STEM = (30, 60, 120)
TREE_TOP = (0, 180, 0)


def norm_to_pixel(x_norm, y_norm, play_size, play_x, play_y):
    """Norm [-1, 1] -> Pixel im Spielbereich. y: positiv = oben, negativ = unten (wie Rohwerte)."""
    center = play_size // 2
    half = play_size / 2.0
    px = play_x + center + int(x_norm * half)
    # y umdrehen: Rohwert positiv (oben) -> kleiner py (oben); Rohwert negativ (unten) -> größer py (unten)
    py = play_y + center - int(y_norm * half)
    return px, py


def pixel_to_norm(px, py, play_size, play_x, play_y):
    """Pixel -> Norm [-1, 1]. y: oben = positiv, unten = negativ."""
    center = play_size // 2
    half = play_size / 2.0
    cx = play_x + center
    cy = play_y + center
    x_norm = (px - cx) / half
    y_norm = (cy - py) / half  # oben (kleines py) -> positiv
    return x_norm, y_norm


def draw_grass(frame, play_x, play_y, size):
    """Rasen: Rechteck + leichte Streifen-Textur."""
    cv2.rectangle(frame, (play_x, play_y), (play_x + size, play_y + size), GRASS_COLOR, -1)
    cv2.rectangle(frame, (play_x, play_y), (play_x + size, play_y + size), GRASS_DARK, 2)
    # Einfache Streifen
    for i in range(0, size, 15):
        cv2.line(frame, (play_x + i, play_y), (play_x + i, play_y + size), GRASS_DARK, 1)
    for j in range(0, size, 15):
        cv2.line(frame, (play_x, play_y + j), (play_x + size, play_y + j), GRASS_DARK, 1)


def draw_water_can(frame, joystick_pos, play_size, play_x, play_y):
    """Wassereimer an Joystick-Position (normiert)."""
    jx, jy = norm_to_pixel(joystick_pos[0], joystick_pos[1], play_size, play_x, play_y)
    # Eimer: kleines Rechteck + Deckel-Ellipse
    w, h = 24, 28
    x1, y1 = jx - w // 2, jy - h // 2
    x2, y2 = jx + w // 2, jy + h // 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), CAN_COLOR, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), CAN_BORDER, 2)
    cv2.ellipse(frame, (jx, y1), (w // 2 + 2, 6), 0, 0, 360, CAN_BORDER, 2)
    # Tropfen/Glanz
    cv2.circle(frame, (jx - 4, jy - 4), 3, (255, 255, 255), -1)


def draw_sapling(frame, target_pos, play_size, play_x, play_y):
    """Setzling = Zielposition: dünner Stamm + ein Blatt oben."""
    tx, ty = norm_to_pixel(target_pos[0], target_pos[1], play_size, play_x, play_y)
    # Stamm (etwas größer)
    cv2.rectangle(frame, (tx - 3, ty), (tx + 3, ty + 20), SAPLING_STEM, -1)
    # Blatt (hochkante Ellipse, Stamm trifft unten mittig)
    leaf_center = (tx, ty - 8)
    axes = (8, 14)  # breit, hoch
    cv2.ellipse(frame, leaf_center, axes, 0, 0, 360, SAPLING_TOP, -1)
    cv2.ellipse(frame, leaf_center, axes, 0, 0, 360, GRASS_DARK, 1)


def draw_tree(frame, target_pos, play_size, play_x, play_y):
    """Ausgewachsener Baum: dicker Stamm + große Krone."""
    tx, ty = norm_to_pixel(target_pos[0], target_pos[1], play_size, play_x, play_y)
    # Stamm (etwas größer)
    cv2.rectangle(frame, (tx - 7, ty + 5), (tx + 7, ty + 45), TREE_STEM, -1)
    cv2.rectangle(frame, (tx - 7, ty + 5), (tx + 7, ty + 45), GRASS_DARK, 1)
    # Krone
    cv2.circle(frame, (tx, ty - 15), 30, TREE_TOP, -1)
    cv2.circle(frame, (tx, ty - 15), 30, GRASS_DARK, 2)


def draw_scoreboard(frame, score, play_x, play_y, size):
    """Scoreboard: 'Trees saved: N' über dem Spielfeld."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"Trees saved: {score}"
    (tw, th), _ = cv2.getTextSize(text, font, 0.8, 2)
    tx = play_x + (size - tw) // 2
    ty = play_y - 15
    cv2.putText(frame, text, (tx, ty), font, 0.8, (0, 0, 0), 3)
    cv2.putText(frame, text, (tx, ty), font, 0.8, (255, 255, 255), 2)


def draw_joystick_position_text(frame, joystick_pos, x=10, y=None, window_height=WINDOW_H):
    """Rohwerte der normalisierten Joystick-Position (x, y) aus dem Env unverändert anzeigen."""
    if y is None:
        y = window_height - 35
    font = cv2.FONT_HERSHEY_SIMPLEX
    x_val = float(joystick_pos[0]) if hasattr(joystick_pos, "__len__") else float(joystick_pos)
    y_val = float(joystick_pos[1]) if hasattr(joystick_pos, "__len__") and len(joystick_pos) > 1 else 0.0
    text = f"Joystick (norm): x = {x_val:+.3f}  y = {y_val:+.3f}"
    cv2.putText(frame, text, (x, y), font, 0.55, (0, 0, 0), 3)
    cv2.putText(frame, text, (x, y), font, 0.55, (255, 255, 255), 1)


def draw_target_position_text(frame, target_pos, x=10, y=None, window_height=WINDOW_H):
    """Rohwerte der normalisierten Target-/Setzling-Position (x, y) in [-1, 1] anzeigen."""
    if y is None:
        y = window_height - 55
    font = cv2.FONT_HERSHEY_SIMPLEX
    x_val = float(target_pos[0]) if hasattr(target_pos, "__len__") else float(target_pos)
    y_val = float(target_pos[1]) if hasattr(target_pos, "__len__") and len(target_pos) > 1 else 0.0
    text = f"Target (norm):  x = {x_val:+.3f}  y = {y_val:+.3f}"
    cv2.putText(frame, text, (x, y), font, 0.55, (0, 0, 0), 3)
    cv2.putText(frame, text, (x, y), font, 0.55, (255, 255, 255), 1)


def draw_grid(frame, play_size, play_x, play_y):
    """Leichtes Grid im Spielbereich (wie im Joystick-Overlay)."""
    center = play_size // 2
    cx, cy = play_x + center, play_y + center
    cv2.line(frame, (cx, play_y), (cx, play_y + play_size), (80, 120, 80), 1)
    cv2.line(frame, (play_x, cy), (play_x + play_size, cy), (80, 120, 80), 1)


def build_game_frame(
    joystick_pos,
    target_pos,
    score,
    show_tree_at=None,
    window_width=WINDOW_W,
    window_height=WINDOW_H,
    play_size=None,
    play_x=None,
    play_y=None,
    margin=PLAY_MARGIN,
):
    """Einzelnen Spiel-Frame bauen (für Recording oder Anzeige)."""
    if play_size is None:
        play_size = min(PLAY_SIZE, window_width - 2 * margin, window_height - 80)
    if play_x is None:
        play_x = margin
    if play_y is None:
        play_y = margin + 30
    frame = np.ones((window_height, window_width, 3), dtype=np.uint8) * 200
    draw_grass(frame, play_x, play_y, play_size)
    draw_grid(frame, play_size, play_x, play_y)
    if show_tree_at is not None:
        draw_tree(frame, np.array(show_tree_at), play_size, play_x, play_y)
    else:
        draw_sapling(frame, target_pos, play_size, play_x, play_y)
    draw_water_can(frame, joystick_pos, play_size, play_x, play_y)
    draw_scoreboard(frame, score, play_x, play_y, play_size)
    draw_target_position_text(frame, target_pos, x=margin, window_height=window_height)
    draw_joystick_position_text(frame, joystick_pos, x=margin, window_height=window_height)
    return frame


def run_game(
    window_width=WINDOW_W,
    window_height=WINDOW_H,
    use_keyboard=True,
    joystick_callback=None,
):
    """
    Spiel-Loop: Rasen, Setzling, Eimer; Treffer -> Baum, Score++, Reset.

    use_keyboard: True = WASD steuert Eimer.
    joystick_callback: Optional callable() -> (x, y) in [-1,1] (z.B. aus Env).
    """
    play_size = min(PLAY_SIZE, window_width - 2 * PLAY_MARGIN, window_height - 80)
    margin = PLAY_MARGIN
    play_x, play_y = margin, margin + 30  # Platz für Scoreboard

    # Zustand (Setzling nur auf TARGETS_20)
    joystick_pos = np.array([0.0, 0.0], dtype=np.float64)
    target_pos = TARGETS_20[np.random.randint(0, len(TARGETS_20))].copy()
    score = 0
    saved_trees = []  # Liste (x, y) für kurze Anzeige nach Treffer
    show_tree_until = 0  # Frame-Zähler: Baum noch N Frames anzeigen

    cv2.namedWindow("Water a Tree", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Water a Tree", window_width, window_height)

    frame_count = 0
    while True:
        frame = np.ones((window_height, window_width, 3), dtype=np.uint8) * 200

        # Joystick-Input
        if callable(joystick_callback):
            try:
                j = joystick_callback()
                joystick_pos = np.array(j, dtype=np.float64)[:2]
                joystick_pos = np.clip(joystick_pos, -1.0, 1.0)
            except Exception:
                pass
        elif use_keyboard:
            # WASD wird in key-Loop unten verarbeitet
            pass

        # Treffer-Check (nur wenn wir keinen "show tree" Zustand haben)
        if show_tree_until <= 0:
            dist = np.linalg.norm(joystick_pos - target_pos)
            if dist < TARGET_RADIUS:
                score += 1
                saved_trees.append(tuple(target_pos))
                show_tree_until = int(FPS * 1.2)  # 1.2 s Baum anzeigen
                # Neues Ziel (nur aus TARGETS_20)
                target_pos = TARGETS_20[np.random.randint(0, len(TARGETS_20))].copy()
        else:
            show_tree_until -= 1
            if show_tree_until <= 0:
                # Optional: letzten Baum nochmal als "saved" zeigen, dann neuer Setzling
                pass

        # Rasen (Spielfeld)
        draw_grass(frame, play_x, play_y, play_size)
        draw_grid(frame, play_size, play_x, play_y)

        # Letzten geretteten Baum zeichnen (während show_tree_until)
        if show_tree_until > 0 and saved_trees:
            draw_tree(frame, np.array(saved_trees[-1]), play_size, play_x, play_y)
        else:
            # Aktuellen Setzling
            draw_sapling(frame, target_pos, play_size, play_x, play_y)

        # Wassereimer
        draw_water_can(frame, joystick_pos, play_size, play_x, play_y)

        # Scoreboard
        draw_scoreboard(frame, score, play_x, play_y, play_size)

        # Normalisierte Koordinaten (Target + Joystick) in [-1, 1]
        draw_target_position_text(frame, target_pos, x=margin, window_height=window_height)
        draw_joystick_position_text(frame, joystick_pos, x=margin, window_height=window_height)

        # Hinweis
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            frame, "WASD = move can | Q = quit",
            (margin, window_height - 10), font, 0.45, (80, 80, 80), 1
        )

        cv2.imshow("Water a Tree", frame)
        frame_count += 1

        key = cv2.waitKey(1000 // FPS) & 0xFF
        if key == ord("q") or key == 27:
            break
        if use_keyboard:
            step = 0.04
            if key == ord("w"):
                joystick_pos[1] = np.clip(joystick_pos[1] - step, -1.0, 1.0)
            elif key == ord("s"):
                joystick_pos[1] = np.clip(joystick_pos[1] + step, -1.0, 1.0)
            elif key == ord("a"):
                joystick_pos[0] = np.clip(joystick_pos[0] - step, -1.0, 1.0)
            elif key == ord("d"):
                joystick_pos[0] = np.clip(joystick_pos[0] + step, -1.0, 1.0)

    cv2.destroyAllWindows()
    print(f"Game over. Trees saved: {score}")


def run_game_with_env_recorded(
    model_path,
    record_path,
    n_episodes=20,
    window_width=WINDOW_W,
    window_height=WINDOW_H,
    fps=FPS,
    show_window=False,
):
    """
    ThumbReach-Env mit trainierter Policy spielen lassen, Wasser-a-Tree-Visualisierung
    aufzeichnen. Joystick = Eimer, Target = Setzling; Treffer = Baum, dann nächste Runde.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    import gymnasium as gym
    from stable_baselines3 import PPO

    import envs  # Registriert ThumbReach-v0

    env = gym.make(
        "ThumbReach-v0",
        normalize_act=True,
        frame_skip=10,
    )
    model = PPO.load(model_path)
    unwrapped = env.unwrapped

    play_size = min(PLAY_SIZE, window_width - 2 * PLAY_MARGIN, window_height - 80)
    margin = PLAY_MARGIN
    play_x, play_y = margin, margin + 30

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(record_path, fourcc, fps, (window_width, window_height))
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter konnte nicht geöffnet werden: {record_path}")

    if show_window:
        cv2.namedWindow("Water a Tree", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Water a Tree", window_width, window_height)

    total_score = 0
    frames_tree_show = int(fps * 1.2)

    for episode in range(n_episodes):
        obs, info = env.reset()
        obs_dict = unwrapped.get_obs_dict(unwrapped.sim)
        joystick_pos = np.array(obs_dict.get("joystick_2d", [0.0, 0.0]), dtype=np.float64)
        target_pos = unwrapped.target_pos.copy()
        episode_score = 0
        done = False

        while not done:
            frame = build_game_frame(
                joystick_pos, target_pos, total_score + episode_score,
                window_width=window_width, window_height=window_height,
                play_size=play_size, play_x=play_x, play_y=play_y, margin=margin,
            )
            writer.write(frame)

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            obs_dict = unwrapped.get_obs_dict(unwrapped.sim)
            joystick_pos = np.array(obs_dict.get("joystick_2d", [0.0, 0.0]), dtype=np.float64)
            target_pos = unwrapped.target_pos.copy()

            if done and info.get("solved", False):
                episode_score += 1
                # Baum kurz anzeigen
                for _ in range(frames_tree_show):
                    frame = build_game_frame(
                        joystick_pos, target_pos, total_score + episode_score,
                        show_tree_at=target_pos,
                        window_width=window_width, window_height=window_height,
                        play_size=play_size, play_x=play_x, play_y=play_y, margin=margin,
                    )
                    writer.write(frame)
                    if show_window:
                        cv2.imshow("Water a Tree", frame)
                        cv2.waitKey(1000 // fps)
                break

            if show_window:
                cv2.imshow("Water a Tree", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    writer.release()
                    env.close()
                    return total_score + episode_score

        total_score += episode_score
        print(f"Runde {episode + 1}/{n_episodes} – Bäume diese Runde: {episode_score}, Gesamt: {total_score}")

    writer.release()
    env.close()
    if show_window:
        cv2.destroyAllWindows()
    print(f"Aufnahme fertig. Gesamt gerettete Bäume: {total_score}. Video: {record_path}")
    return total_score


def main():
    parser = argparse.ArgumentParser(
        description="Water a Tree – Joystick = water can, targets = saplings, score = trees saved."
    )
    parser.add_argument("--width", type=int, default=WINDOW_W, help="Window width")
    parser.add_argument("--height", type=int, default=WINDOW_H, help="Window height")
    parser.add_argument("--no-keyboard", action="store_true", help="Disable WASD (use joystick_callback only)")
    parser.add_argument("--model", type=str, default=None, help="Pfad zu trainiertem PPO-Modell (.zip) für Joystick-Steuerung")
    parser.add_argument("--episodes", type=int, default=20, help="Anzahl Runden bei --model + --record")
    parser.add_argument("--record", type=str, default=None, help="Video-Ausgabepfad (z.B. water_a_tree_20ep.mp4)")
    parser.add_argument("--show", action="store_true", help="Fenster anzeigen während der Aufnahme (--model + --record)")
    args = parser.parse_args()

    if args.model and args.record:
        run_game_with_env_recorded(
            model_path=args.model,
            record_path=args.record,
            n_episodes=args.episodes,
            window_width=args.width,
            window_height=args.height,
            show_window=args.show,
        )
        return

    run_game(
        window_width=args.width,
        window_height=args.height,
        use_keyboard=not args.no_keyboard,
    )


def get_joystick_from_env(env):
    """
    Callback für run_game(joystick_callback=...), wenn Joystick aus dem
    ThumbReach-Environment kommen soll (wie in create_video.py).
    Verwendung:
        import gymnasium as gym
        import envs
        env = gym.make("ThumbReach-v0")
        obs, _ = env.reset()
        def cb():
            d = env.unwrapped.get_obs_dict(env.unwrapped.sim)
            return d.get("joystick_2d", [0, 0])
        run_game(use_keyboard=False, joystick_callback=cb)
    """
    def callback():
        obs_dict = env.unwrapped.get_obs_dict(env.unwrapped.sim)
        return obs_dict.get("joystick_2d", np.array([0.0, 0.0]))
    return callback


if __name__ == "__main__":
    main()
