 #!/usr/bin/env python3
  """Print Isaac Gym DOF order for the HoST pi_plus_ground task.                                                                                                                                                                      
                                                                                                                                                                                                                                      
  Run inside the HoST pixi environment, from the legged_gym repo root:                                                                                                                                                                
                                                                                                                                                                                                                                      
      pixi run python legged_gym/scripts/print_dof_names.py --task pi_plus_ground --num_envs 1 --headless
                                                                                                                                                                                                                                      
  If --task is omitted it defaults to pi_plus_ground.
  """                                                                                                                                                                                                                                 

  import isaacgym  # MUST be imported before torch                                                                                                                                                                                    
  import torch  # noqa: F401   
                                                                                                                                                                                                                                      
  from legged_gym.envs import *  # noqa: F401, F403  -- registers tasks                                                                                                                                                               
  from legged_gym.utils import task_registry, get_args
                                                                                                                                                                                                                                      

  def main():
      args = get_args()
      if not args.task:
          args.task = "pi_plus_ground"                                                                                                                                                                                                
      args.headless = True      
                                                                                                                                                                                                                                      
      env_cfg, _ = task_registry.get_cfgs(name=args.task)                                                                                                                                                                             
                                  
      # minimise everything so init is fast and cheap                                                                                                                                                                                 
      env_cfg.env.num_envs = 1    
      env_cfg.terrain.num_rows = 1                                                                                                                                                                                                    
      env_cfg.terrain.num_cols = 1
      env_cfg.terrain.curriculum = False                                                                                                                                                                                              
      env_cfg.noise.add_noise = False
      env_cfg.curriculum.pull_force = False                                                                                                                                                                                           
      env_cfg.env.test = True                                                                                                                                                                                                         
                                     
      env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)                                                                                                                                                     

      print(f"\n=== Isaac Gym DOF order for task '{args.task}' ===")                                                                                                                                                                  
      print(f"num_dof = {env.num_dof}")
      for i, name in enumerate(env.dof_names):                                                                                                                                                                                        
          print(f"  [{i:2d}] {name}")
      print("=== end ===\n")                                                                                                                                                                                                          
                                        
                                                                                                                                                                                                                                      
  if __name__ == "__main__":
      main()

