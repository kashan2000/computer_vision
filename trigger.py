from counts import (
    toe_tap_trigger, push_pull_trigger, push_pull_left_trigger, 
    push_pull_right_trigger, inside_tap_trigger, inside_outside_right_trigger, 
    inside_outside_left_trigger, v_push_pull_trigger, roll_across_trigger
)

# temp__ = [
#     {
#         "type": "toe",
#         "func": toe_tap_trigger
#     }
# ]


def get_trigger(drill_type, detection_results_dict, pose_results_dict, frame_count):

    prev_count = 0
    if frame_count in detection_results_dict and frame_count in pose_results_dict:
        merged_data = {
            'frame_count': frame_count,
            'detection': detection_results_dict[frame_count],
            'pose': pose_results_dict[frame_count]
        }

        if drill_type == "toe_taps":
            count, trigger = toe_tap_trigger(merged_data, frame_count)
        elif drill_type == "push_pull":
            count, trigger = push_pull_trigger(merged_data, frame_count)
        elif drill_type == "push_pull_left":
            count, trigger = push_pull_left_trigger(merged_data, frame_count)
        elif drill_type == "push_pull_right":
            count, trigger = push_pull_right_trigger(merged_data, frame_count)
        elif drill_type == "inside_taps":
            count, trigger = inside_tap_trigger(merged_data, frame_count)
        elif drill_type == "inside_outside_right":
            count, trigger = inside_outside_right_trigger(merged_data, frame_count)
        elif drill_type == "inside_outside_left":
            count, trigger = inside_outside_left_trigger(merged_data, frame_count)
        elif drill_type == "v_push_pull":
            count, trigger = v_push_pull_trigger(merged_data, frame_count)
        elif drill_type == "roll_across":
            count, trigger = roll_across_trigger(merged_data, frame_count)
        else:
            count, trigger = 0, False

        if count > 0:
            prev_count = count    

        print(f"Count is {prev_count} Trigger is {trigger} Frame Count is {frame_count}")
        return prev_count, trigger
   
    return prev_count, False
