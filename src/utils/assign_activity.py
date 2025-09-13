import yaml

def assign_activity(label : str):
    with open('./config/activities.yaml', 'r') as file:
        activities_dict = yaml.safe_load(file)

    # identificando actividad
    activity = activities_dict['config'][label]

    return activities_dict['activities'][activity]