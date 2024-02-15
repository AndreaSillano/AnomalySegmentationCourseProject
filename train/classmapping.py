rgb_to_class_mapping = { 
    (64, 128, 64): ('Animal', 0, 19), 
    (192, 0, 128): ('Archway', 1, 2), 
    (0, 128, 192): ('Bicyclist', 2, 12), 
    (0, 128, 64): ('Bridge', 3,2), 
    (128, 0, 0): ('Building', 4,2), 
    (64, 0, 128): ('Car', 5,13), 
    (64, 0, 192): ('CartLuggagePram', 6, 13), 
    (192, 128, 64): ('Child', 7,11), 
    (192, 192, 128): ('Column_Pole', 8,5), 
    (64, 64, 128): ('Fence', 9,4), 
    (128, 0, 192): ('LaneMkgsDriv', 10,0), 
    (192, 0, 64): ('LaneMkgsNonDriv', 11,0), 
    (128, 128, 64): ('Misc_Text', 12,19), 
    (192, 0, 192): ('MotorcycleScooter', 13,17), 
    (128, 64, 64): ('OtherMoving', 14,19), 
    (64, 192, 128): ('ParkingBlock', 15,19), 
    (64, 64, 0): ('Pedestrian', 16,11), 
    (128, 64, 128): ('Road', 17,0), 
    (128, 128, 192): ('RoadShoulder', 18,1), 
    (0, 0, 192): ('Sidewalk', 19,1), 
    (192, 128, 128): ('SignSymbol', 20,7), 
    (128, 128, 128): ('Sky', 21,10), 
    (64, 128, 192): ('SUVPickupTruck', 22,14), 
    (0, 0, 64): ('TrafficCone', 23,7), 
    (0, 64, 64): ('TrafficLight', 24,6), 
    (192, 64, 128): ('Train', 25,16), 
    (128, 128, 0): ('Tree', 26,8), 
    (192, 128, 192): ('Truck_Bus', 27,15), 
    (64, 0, 64): ('Tunnel', 28,2), 
    (192, 192, 0): ('VegetationMisc', 29,8), 
    (0, 0, 0): ('Void', 30,19), 
    (64, 192, 0): ('Wall', 31,3) 
}

def get_label_cityscapes(rgb)->int:
    if (rgb[0], rgb[1],rgb[2]) in rgb_to_class_mapping:
      return rgb_to_class_mapping[(rgb[0], rgb[1],rgb[2])][2]
    else: 
      return 19