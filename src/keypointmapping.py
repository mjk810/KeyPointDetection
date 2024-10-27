
#TODO this should be a data class
class KeyPointMapping():
    KEY_POINT_NAMES = ['nose', 'left eye', 'right eye', 'left ear', 
                       'right ear', 'left shoulder', 'right shoulder', 
                       'left elbow', 'right elbow', 'left wrist', 
                       'right wrist', 'left hip', 'right hip', 'left knee', 
                       'right knee', 'left ankle', 'right ankle'
                       ]
    LINES = [('right knee', 'right ankle'),
             ('left knee', 'left ankle'),
             ('left shoulder', 'right shoulder'),
             ('left shoulder','left elbow'),
             ('left elbow','left wrist'),
             ('right shoulder','right elbow'),
             ('right elbow','right wrist'),
             ('left hip', 'right hip'),
             ('right shoulder','right hip'),
             ('left shoulder', 'left hip'),
             ('left eye', 'right eye'),
             ('left eye', 'left ear'),
             ('right eye', 'right ear'),
             ('left eye', 'nose'),
             ('right eye', 'nose'),
             ('nose', 'right shoulder'),
             ('nose', 'left shoulder'),
             ('right hip', 'right knee'),
             ('left hip', 'left knee')
             ]