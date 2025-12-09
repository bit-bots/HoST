# Isaac Gym zuerst importieren
import isaacgym
from isaacgym import gymapi, gymutil

# kleine Nutzung, damit der Editor nicht meckert
_ = gymapi.SimParams()
_ = gymutil.parse_arguments()

import torch
from legged_gym.envs import task_registry
from legged_gym.utils import get_args


def main():
    args = get_args()

    # sicherstellen, dass wir die X02 Umgebung nutzen
    args.task = "x02_ground"
    args.num_envs = 1
    args.headless = False

    # Umgebung erstellen
    env, env_cfg = task_registry.make_env(name=args.task, args=args)

    # Zugriff auf Isaac Gym Handles
    gym = env.gym
    viewer = env.viewer

    # Aktionsvektor, zuerst alles Null
    actions = torch.zeros(
        env.num_envs,
        env_cfg.env.num_actions,
        device=env.device,
    )

    # DoF Namen und Indizes holen
    dof_names = list(env.dof_names)

    def idx(name):
        return dof_names.index(name)

    idx_L_hip_pitch = idx("L_hip_pitch")
    idx_L_knee_pitch = idx("L_knee_pitch")
    idx_L_ankle_pitch = idx("L_ankle_pitch")

    idx_R_hip_pitch = idx("R_hip_pitch")
    idx_R_knee_pitch = idx("R_knee_pitch")
    idx_R_ankle_pitch = idx("R_ankle_pitch")

    # wie stark ein Tastendruck wirken soll
    joint_step_rad = 5.0 * 3.14159265 / 180.0
    action_scale = env_cfg.control.action_scale
    action_step = joint_step_rad / action_scale

    # Tastatur Events abonnieren
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_T, "L_HIP_P_UP")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_G, "L_HIP_P_DOWN")

    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_Y, "L_KNEE_UP")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_H, "L_KNEE_DOWN")

    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_U, "L_ANKLE_UP")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_J, "L_ANKLE_DOWN")

    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_I, "R_HIP_P_UP")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_K, "R_HIP_P_DOWN")

    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_O, "R_KNEE_UP")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_L, "R_KNEE_DOWN")

    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_P, "R_ANKLE_UP")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_SEMICOLON, "R_ANKLE_DOWN")

    print()
    print("X02 Viewer mit Gelenksteuerung")
    print("Linkes Bein: T G, Y H, U J")
    print("Rechtes Bein: I K, O L, P Semikolon")
    print("Strg plus C im Terminal beendet das Programm")
    print()

    while True:
        # Viewer Schließen erkennen
        if gym.query_viewer_has_closed(viewer):
            break

        # Tastaturevents auslesen
        for evt in gym.query_viewer_action_events(viewer):
            if evt.value == 0:
                continue  # nur Tastendruck, nicht Loslassen

            if evt.action == "L_HIP_P_UP":
                actions[:, idx_L_hip_pitch] += action_step
            elif evt.action == "L_HIP_P_DOWN":
                actions[:, idx_L_hip_pitch] -= action_step

            elif evt.action == "L_KNEE_UP":
                actions[:, idx_L_knee_pitch] += action_step
            elif evt.action == "L_KNEE_DOWN":
                actions[:, idx_L_knee_pitch] -= action_step

            elif evt.action == "L_ANKLE_UP":
                actions[:, idx_L_ankle_pitch] += action_step
            elif evt.action == "L_ANKLE_DOWN":
                actions[:, idx_L_ankle_pitch] -= action_step

            elif evt.action == "R_HIP_P_UP":
                actions[:, idx_R_hip_pitch] += action_step
            elif evt.action == "R_HIP_P_DOWN":
                actions[:, idx_R_hip_pitch] -= action_step

            elif evt.action == "R_KNEE_UP":
                actions[:, idx_R_knee_pitch] += action_step
            elif evt.action == "R_KNEE_DOWN":
                actions[:, idx_R_knee_pitch] -= action_step

            elif evt.action == "R_ANKLE_UP":
                actions[:, idx_R_ankle_pitch] += action_step
            elif evt.action == "R_ANKLE_DOWN":
                actions[:, idx_R_ankle_pitch] -= action_step

        # Aktionen sinnvoll begrenzen
        actions.clamp_(-1.0, 1.0)

        # ein Simulationsschritt mit aktuellen Aktionen
        obs, privileged_obs, rew, done, info = env.step(actions)

        # kein separates render nötig, das macht env intern

    print("Viewer geschlossen")


if __name__ == "__main__":
    main()
