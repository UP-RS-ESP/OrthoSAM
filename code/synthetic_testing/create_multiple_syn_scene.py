import subprocess

for radi in [8,16,32,64]:
    print('Creating scene with min radi: ', radi)
    subprocess.run(["python", 'code/synthetic_testing/create_syn_scene.py', f'{radi}'])
print('Creatomg colored scene')
subprocess.run(["python", 'code/synthetic_testing/create_syn_scene.py', '64', 'True'])
print('Add noise')
subprocess.run(["python", 'code/synthetic_testing/add_noise_syn_scene.py'])