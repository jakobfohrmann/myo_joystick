"""Video Generation Script für Thumb Reach Training

Lädt ein trainiertes Modell und erstellt ein Video der Episodes.
Verwendet MyoSuite Renderer und cv2.VideoWriter.
"""
import os
import sys
import argparse
import numpy as np
import cv2

# Füge project Ordner zum Python-Pfad hinzu
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import gymnasium as gym
from stable_baselines3 import PPO

# Registriert Environment ThumbReach-v0
import envs  

ENV_NAME = "ThumbReach-v0"


def draw_joystick_overlay(frame, joystick_pos, target_pos, size=200, margin=20):
    """
    Zeichnet eine 2D-Joystick-Visualisierung als Overlay auf das Frame.

    Args:
        frame: Videoframe (NumPy-Array, BGR-Format)
        joystick_pos: Aktuelle Joystick-Position [x, y], normalisiert auf [-1, 1]
        target_pos: Zielposition [x, y], normalisiert auf [-1, 1]
        size: Größe des Visualisierungs-Quadrats in Pixeln
        margin: Abstand zu den Rändern in Pixeln

    Returns:
        Frame mit hinzugefügtem Overlay
    """
    h, w = frame.shape[:2]
    
    # Position: oben rechte Ecke
    overlay_x = w - size - margin
    overlay_y = margin
    
    # Overlay-Bereich erstellen
    overlay = frame[overlay_y:overlay_y+size, overlay_x:overlay_x+size].copy()
    
    # Hintergrund zeichnen
    cv2.rectangle(overlay, (0, 0), (size, size), (240, 240, 240), -1)
    cv2.rectangle(overlay, (0, 0), (size, size), (200, 200, 200), 2)
    
    # Grid zeichnen
    center = size // 2
    cv2.line(overlay, (center, 0), (center, size), (180, 180, 180), 1)
    cv2.line(overlay, (0, center), (size, center), (180, 180, 180), 1)
    
    # Koordinatenbeschriftungen zeichnen
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    thickness = 1
    cv2.putText(overlay, "1", (size - 15, center - 5), font, font_scale, (100, 100, 100), thickness)
    cv2.putText(overlay, "-1", (5, center - 5), font, font_scale, (100, 100, 100), thickness)
    cv2.putText(overlay, "1", (center + 5, 15), font, font_scale, (100, 100, 100), thickness)
    cv2.putText(overlay, "-1", (center + 5, size - 5), font, font_scale, (100, 100, 100), thickness)
    
    # Normalisierte Koordinaten [-1, 1] in Pixelkoordinaten umrechnen
    # Ohne Y-Flip: x_pixel = (x_norm + 1) * size/2, y_pixel = (y_norm + 1) * size/2
    def norm_to_pixel(x_norm, y_norm):
        x_pixel = int((x_norm + 1) * size / 2)
        y_pixel = int((y_norm + 1) * size / 2)
        return x_pixel, y_pixel
    
    # Zeichne Target-Position (roter Kreis)
    target_x, target_y = norm_to_pixel(target_pos[0], target_pos[1])
    cv2.circle(overlay, (target_x, target_y), 8, (0, 0, 255), -1)  # rot gefüllter Kreis
    cv2.circle(overlay, (target_x, target_y), 10, (0, 0, 200), 2)  # roter Rahmen
    
    # Zeichne Joystick-Position (blauer Kreis)
    # Joystick-Position ohne Y-Flip (gleiche Orientierung wie Targets/Heatmap)
    joy_x, joy_y = norm_to_pixel(joystick_pos[0], joystick_pos[1])
    cv2.circle(overlay, (joy_x, joy_y), 6, (255, 0, 0), -1)  # blau gefüllter Kreis
    cv2.circle(overlay, (joy_x, joy_y), 8, (200, 0, 0), 2)  # blauer Rahmen
    
    # Linie vom Joystick zum Ziel zeichnen
    cv2.line(overlay, (joy_x, joy_y), (target_x, target_y), (100, 100, 100), 1)
    
    # Legende zeichnen
    legend_y = size - 50
    cv2.circle(overlay, (10, legend_y), 5, (0, 0, 255), -1)
    cv2.putText(overlay, "Target", (20, legend_y + 5), font, 0.35, (0, 0, 0), thickness)
    
    cv2.circle(overlay, (10, legend_y + 15), 5, (255, 0, 0), -1)  # Blue (BGR)
    cv2.putText(overlay, "Joystick", (20, legend_y + 20), font, 0.35, (0, 0, 0), thickness)
    
    # Overlay zurück ins Frame kopieren
    frame[overlay_y:overlay_y+size, overlay_x:overlay_x+size] = overlay
    
    return frame


def create_video(
    model_path: str,
    output_path: str = "video.mp4",
    n_episodes: int = 3,
    fps: int = 30,
    width: int = 640,
    height: int = 480,
    camera_id: int = -1,  # -1 = freie Kamera
):
    """
    Erstellt Video von trainiertem Modell mit MyoSuite Renderer und cv2.VideoWriter
    
    Args:
        model_path: Pfad zum trainierten Modell (.zip)
        output_path: Ausgabe-Pfad für Video
        n_episodes: Anzahl Episodes für Video
        fps: Frames pro Sekunde
        width: Video-Breite
        height: Video-Höhe
        camera_id: Kamera-ID (-1 für free camera, oder Name/String)
    """
    print("=" * 60)
    print("Loading Model and Environment")
    print("=" * 60)
    
    # Environment Config (muss mit Training übereinstimmen)
    env_config = {
        "target_range": {
            "x": (-1.0, 1.0),
            "y": (-1.0, 1.0),
        },
        "far_th": 0.5,
        "early_term_threshold": 0.15,
        "early_term_steps": 100,
        "weighted_reward_keys": {
            "reach": 1.0,
            "bonus": 40.0,
        },
    }
    
    # Environment erstellen
    print("Creating environment...")
    env = gym.make(ENV_NAME, **env_config)
    env_unwrapped = env.unwrapped
    
    # Zugriff auf MyoSuite Renderer (falls verfügbar)
    # BaseV0 hat self.sim.renderer
    renderer = None
    use_myosuite_renderer = False
    
    if hasattr(env_unwrapped, 'sim') and hasattr(env_unwrapped.sim, 'renderer'):
        try:
            renderer = env_unwrapped.sim.renderer
            if hasattr(renderer, 'render_offscreen'):
                use_myosuite_renderer = True
                print("Using MyoSuite Renderer from environment")
            else:
                print("Renderer found but no render_offscreen method, using fallback")
                renderer = None
        except Exception as e:
            print(f"Could not use MyoSuite renderer: {e}, using fallback")
            renderer = None
    
    # Fallback: Direkter MuJoCo Renderer
    if renderer is None:
        import mujoco
        sim = env_unwrapped.sim
        print("Using direct MuJoCo renderer as fallback")
        # Renderer wird später initialisiert, da wir sim.model.ptr brauchen
    
    # Model laden
    print(f"Loading model from {model_path}")
    model = PPO.load(model_path)
    
    # Video Writer initialisieren (wie in user-in-the-box)
    print(f"Initializing video writer: {output_path}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )
    
    if not video_writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_path}")
    
    print("=" * 60)
    print("Generating Video")
    print("=" * 60)
    print(f"Episodes: {n_episodes}")
    print(f"Output: {output_path}")
    print(f"Resolution: {width}x{height}, FPS: {fps}")
    print("=" * 60)
    
    # Statistics
    episode_rewards = []
    episode_lengths = []
    total_frames = 0
    
    # Fallback Renderer initialisieren (falls nötig)
    fallback_renderer = None
    if not use_myosuite_renderer:
        import mujoco
        sim = env_unwrapped.sim
        # Verwende sim.model.ptr für MuJoCo Renderer
        try:
            # Versuche auf native MuJoCo model zuzugreifen
            if hasattr(sim.model, 'ptr'):
                fallback_renderer = mujoco.Renderer(sim.model.ptr, width=width, height=height)
            else:
                # Fallback: Lade XML direkt
                xml_path = os.path.join(project_root, "xml", "controller_with_hand.xml")
                native_model = mujoco.MjModel.from_xml_path(xml_path)
                fallback_renderer = mujoco.Renderer(native_model, width=width, height=height)
                # Wir müssen native_data synchronisieren
                native_data = mujoco.MjData(native_model)
        except Exception as e:
            print(f"Could not create fallback renderer: {e}")
            raise
    
    for episode in range(n_episodes):
        print(f"\nEpisode {episode + 1}/{n_episodes}")
        obs, info = env.reset()
        # Debug: zeige Target nach reset()
        try:
            target_pos = getattr(env_unwrapped, "target_pos", None)
            if target_pos is not None:
                # Wenn Env FIXED_TARGETS/TARGET_NAMES hat, gib Name aus (sonst nur Position)
                name = None
                if hasattr(env_unwrapped, "FIXED_TARGETS") and hasattr(env_unwrapped, "TARGET_NAMES"):
                    for i, t in enumerate(env_unwrapped.FIXED_TARGETS):
                        if np.allclose(target_pos, t):
                            name = env_unwrapped.TARGET_NAMES[i]
                            break
                if name is None:
                    print(f"  Target: Position {target_pos}")
                else:
                    print(f"  Target: {name}, Position: {target_pos}")
        except Exception:
            pass
        done = False
        episode_reward = 0
        episode_frames = 0
        last_joystick_pos = None
        last_target_pos = None
        
        step = 0
        while not done:
            # Action vom Modell
            action, _ = model.predict(obs, deterministic=True)
            
            # Step
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            
            # Frame rendern
            if use_myosuite_renderer:
                # MyoSuite Renderer verwenden
                frame = renderer.render_offscreen(
                    width=width,
                    height=height,
                    rgb=True,
                    camera_id=camera_id
                )
            else:
                # Fallback: Direkter MuJoCo Renderer
                sim = env_unwrapped.sim
                
                # Synchronisiere native_data mit sim.data (falls nötig)
                if fallback_renderer is not None and 'native_data' in locals():
                    nq = min(len(native_data.qpos), len(sim.data.qpos))
                    nv = min(len(native_data.qvel), len(sim.data.qvel))
                    native_data.qpos[:nq] = sim.data.qpos[:nq]
                    native_data.qvel[:nv] = sim.data.qvel[:nv]
                    if hasattr(sim.data, 'ctrl') and sim.data.ctrl is not None:
                        nu = min(len(native_data.ctrl), len(sim.data.ctrl))
                        native_data.ctrl[:nu] = sim.data.ctrl[:nu]
                    
                    mujoco.mj_forward(native_model, native_data)
                    fallback_renderer.update_scene(native_data, camera=camera_id)
                else:
                    # Verwende sim direkt
                    if hasattr(sim.model, 'ptr') and hasattr(sim.data, 'ptr'):
                        mujoco.mj_forward(sim.model.ptr, sim.data.ptr)
                        fallback_renderer.update_scene(sim.data.ptr, camera=camera_id)
                    else:
                        # Letzter Fallback: Verwende native_model/native_data
                        if 'native_model' not in locals():
                            import mujoco
                            xml_path = os.path.join(project_root, "xml", "controller_with_hand.xml")
                            native_model = mujoco.MjModel.from_xml_path(xml_path)
                            native_data = mujoco.MjData(native_model)
                            fallback_renderer = mujoco.Renderer(native_model, width=width, height=height)
                        
                        nq = min(len(native_data.qpos), len(sim.data.qpos))
                        nv = min(len(native_data.qvel), len(sim.data.qvel))
                        native_data.qpos[:nq] = sim.data.qpos[:nq]
                        native_data.qvel[:nv] = sim.data.qvel[:nv]
                        if hasattr(sim.data, 'ctrl') and sim.data.ctrl is not None:
                            nu = min(len(native_data.ctrl), len(sim.data.ctrl))
                            native_data.ctrl[:nu] = sim.data.ctrl[:nu]
                        
                        mujoco.mj_forward(native_model, native_data)
                        fallback_renderer.update_scene(native_data, camera=camera_id)
                
                frame = fallback_renderer.render()
            
            # Konvertiere RGB zu BGR für OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Get joystick and target positions from observation dictionary
            try:
                # Access observation dictionary from environment
                if hasattr(env_unwrapped, 'get_obs_dict'):
                    obs_dict = env_unwrapped.get_obs_dict(env_unwrapped.sim)
                    joystick_pos = obs_dict.get('joystick_2d', np.array([0.0, 0.0]))
                    target_pos = obs_dict.get('target_pos', np.array([0.0, 0.0]))
                else:
                    # Fallback: try to get from environment attributes or info
                    if hasattr(env_unwrapped, 'target_pos'):
                        target_pos = env_unwrapped.target_pos
                    else:
                        target_pos = info.get('target_pos', np.array([0.0, 0.0])) if isinstance(info, dict) else np.array([0.0, 0.0])
                    
                    # Try to get joystick position from observation or calculate it
                    if hasattr(env_unwrapped, 'joystick_rx_joint_id') and hasattr(env_unwrapped, 'joystick_ry_joint_id'):
                        if env_unwrapped.joystick_rx_joint_id is not None and env_unwrapped.joystick_ry_joint_id is not None:
                            sim = env_unwrapped.sim
                            joystick_angles = np.array([
                                sim.data.qpos[env_unwrapped.joystick_rx_joint_id],
                                sim.data.qpos[env_unwrapped.joystick_ry_joint_id]
                            ])
                            rx_norm = (joystick_angles[0] - env_unwrapped.joystick_rx_center) / env_unwrapped.joystick_rx_span if env_unwrapped.joystick_rx_span > 0 else 0
                            ry_norm = (joystick_angles[1] - env_unwrapped.joystick_ry_center) / env_unwrapped.joystick_ry_span if env_unwrapped.joystick_ry_span > 0 else 0
                            joystick_pos = np.array([rx_norm, ry_norm])
                        else:
                            joystick_pos = np.array([0.0, 0.0])
                    else:
                        joystick_pos = info.get('joystick_2d', np.array([0.0, 0.0])) if isinstance(info, dict) else np.array([0.0, 0.0])
                
                # Ensure we have numpy arrays
                if not isinstance(joystick_pos, np.ndarray):
                    joystick_pos = np.array(joystick_pos)
                if not isinstance(target_pos, np.ndarray):
                    target_pos = np.array(target_pos)
                
                # Ensure 2D arrays (take first 2 elements if needed)
                if len(joystick_pos) > 2:
                    joystick_pos = joystick_pos[:2]
                if len(target_pos) > 2:
                    target_pos = target_pos[:2]
                
                # Merke letzte Positionen für Logging nach Episode
                last_joystick_pos = joystick_pos
                last_target_pos = target_pos
                
                # Draw joystick overlay on frame
                frame_bgr = draw_joystick_overlay(frame_bgr, joystick_pos, target_pos)
            except Exception as e:
                # If overlay fails, continue without it
                print(f"Warning: Could not draw joystick overlay: {e}")
            
            # Frame direkt zum Video hinzufügen (wie in user-in-the-box)
            video_writer.write(frame_bgr)
            episode_frames += 1
            total_frames += 1
            
            step += 1
            if step % 50 == 0:
                print(f"  Step {step}, Reward: {episode_reward:.2f}")
        
        print(f"  Episode finished! Total reward: {episode_reward:.2f}, Steps: {step}, Frames: {episode_frames}")
        # Ausgabe der finalen Joystick-Position zusammen mit dem Ziel
        if last_joystick_pos is not None:
            print(f"    Final joystick position: {np.array2string(last_joystick_pos, precision=3)}")
        if last_target_pos is not None:
            print(f"    Target position: {np.array2string(last_target_pos, precision=3)}")
        episode_rewards.append(episode_reward)
        episode_lengths.append(step)
        
        # Optional: Pause zwischen Episodes (schwarze Frames)
        if episode < n_episodes - 1:
            pause_frames = int(fps * 1.0)  # 1 Sekunde Pause
            black_frame = np.zeros((height, width, 3), dtype=np.uint8)
            for _ in range(pause_frames):
                video_writer.write(black_frame)
                total_frames += 1
    
    # Video Writer schließen
    video_writer.release()
    
    # Renderer schließen
    if fallback_renderer is not None:
        fallback_renderer.close()
    env.close()
    
    # Statistics ausgeben
    print("\n" + "=" * 60)
    print("Video Generation Complete")
    print("=" * 60)
    print(f"   Video saved: {output_path}")
    print(f"   Episodes: {n_episodes}")
    print(f"   Average reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"   Average length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f} steps")
    print(f"   Total frames: {total_frames}")
    print(f"   Video length: {total_frames / fps:.1f} seconds")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Create video from trained Thumb Reach model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Grundlegende Verwendung mit dem finalen Modell
  python models/create_video.py --model logs/thumb_reach_*/final_model.zip --episodes 5

  # Mit individueller Ausgabe und Episoden
  python models/create_video.py \\
      --model logs/thumb_reach_*/final_model.zip \\
      --episodes 10 \\
      --output my_video.mp4

  # Checkpoint verwenden
  python models/create_video.py \\
      --model logs/thumb_reach_*/checkpoint_50000.zip \\
      --episodes 3 \\
      --fps 24
        """
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model (.zip file)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="thumb_reach_video.mp4",
        help="Output video path (default: thumb_reach_video.mp4)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes to record (default: 3)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second (default: 30)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Video width (default: 640)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Video height (default: 480)",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="-1",
        help="Camera ID or name (default: -1 for free camera)",
    )
    
    args = parser.parse_args()
    
    # Parse camera_id
    try:
        camera_id = int(args.camera)
    except ValueError:
        camera_id = args.camera  # String Kamera Name
    
    create_video(
        model_path=args.model,
        output_path=args.output,
        n_episodes=args.episodes,
        fps=args.fps,
        width=args.width,
        height=args.height,
        camera_id=camera_id,
    )


if __name__ == "__main__":
    main()
