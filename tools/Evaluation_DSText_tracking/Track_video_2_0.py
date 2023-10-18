#!/usr/bin/env python
# -*- coding: utf-8 -*-

#File: TL_video_2_0.py
#Version: 2.0
#Date: 2019-12-29
#Description: Evaluation script that computes Text Localization with the py-motmetrics library

from io import StringIO
from io import BytesIO
import rrc_evaluation_funcs_1_1 as rrc_evaluation_funcs
import importlib
import xml.etree.ElementTree as ET
import re
import math
from tqdm import tqdm
def evaluation_imports():
    """
    evaluation_imports: Dictionary ( key = module name , value = alias  )  with python modules used in the evaluation. 
    """    
    return {
                'lxml':'lxml',
                'Polygon':'plg',
                'Polygon.Utils':'Utils',
                'numpy':'np',
                'motmetrics':'mm'
            }

def default_evaluation_params():
    """
    default_evaluation_params: Default parameters to use for the validation and evaluation.
    """        
    return {
                'GT_SAMPLE_NAME_2_ID':'Video_([0-9]+)_([0-9]+)_([0-9]+)_GT.xml',
                'DET_SAMPLE_NAME_2_ID':'res_Video_([0-9]+)_([0-9]+)_([0-9]+).xml',
                'DISTANCE_THRESHOLD' : 0.5
            }

def validate_data(gtFilePath, submFilePath,evaluationParams):
    """
    Method validate_data: validates that all files in the results folder are correct (have the correct name contents).
                            Validates also that there are no missing files in the folder.
                            If some error detected, the method raises the error
    """
    
    for module,alias in evaluation_imports().items():
        globals()[alias] = importlib.import_module(module) 
        
    gt = rrc_evaluation_funcs.load_zip_file(gtFilePath, evaluationParams['GT_SAMPLE_NAME_2_ID'])

    subm = rrc_evaluation_funcs.load_zip_file(submFilePath, evaluationParams['DET_SAMPLE_NAME_2_ID'], True)
    
    from lxml import etree
	
    f = StringIO('''
        <xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
          <xs:element name="Frames">
            <xs:complexType>
              <xs:sequence>
                <xs:element ref="frame" minOccurs="1" maxOccurs="unbounded" />
              </xs:sequence>
              <xs:attribute name="ID" type="xs:integer" use="optional" />
              <xs:attribute name="video_name" type="xs:string" use="optional" />
              <xs:attribute name="author" type="xs:string" use="optional" />
              <xs:attribute name="comments" type="xs:string" use="optional" />
            </xs:complexType>
          </xs:element>
         <xs:element name="frame">
            <xs:complexType>
              <xs:sequence>
                <xs:element ref="object" minOccurs="0" maxOccurs="unbounded" />
              </xs:sequence>
              <xs:attribute name="ID" type="xs:integer" use="required" />
            </xs:complexType>
          </xs:element>
         <xs:element name="object">
            <xs:complexType>
              <xs:sequence>
                <xs:element ref="Point" minOccurs="4" maxOccurs="unbounded" />
              </xs:sequence>
              <xs:attribute name="ID" type="xs:integer" use="required" />
                <xs:attribute name="Transcription" type="xs:string" use="optional" />
                <xs:attribute name="Language" type="xs:string" use="optional" />
                <xs:attribute name="Mirrored" type="xs:string" use="optional" />
              <xs:attribute name="Quality" type="xs:string" use="optional" />
            </xs:complexType>
          </xs:element>
           <xs:element name="Point">
            <xs:complexType>
              <xs:attribute name="x" type="xs:integer" use="required" />
              <xs:attribute name="y" type="xs:integer" use="required" />
            </xs:complexType>
          </xs:element>
        </xs:schema>
    ''')
    xmlschema_doc = etree.parse(f)
    xmlschema = etree.XMLSchema(xmlschema_doc)    
    
    
    #Validate format of GroundTruth
    #for k in gt:
    #    valid = StringIO(gt[k])
    #    doc = etree.parse(valid)
    #    xmlschema.assertValid(doc)
        #if xmlschema.validate(doc) == False:
        #    raise Exception("The GT XML file ID %s is not valid" %k)

    #Validate format of results
    for k in subm:
        if (k in gt) == False :
            raise Exception("The video ID %s is not present in GT" %k)
        
        valid = BytesIO(subm[k])
        doc = etree.parse(valid)
        try:
            xmlschema.assertValid(doc)
        except Exception as e:
            raise Exception("The XML file of the video ID %s is not valid. Error: %s" %(k,e))
        
        root = ET.fromstring(subm[k])
        objectsDict = {}
        for frame in list(root.iter('frame')):
            frameObjectsDict = {}
            for obj in list(frame.iter('object')):
                if obj.attrib['ID'] in frameObjectsDict:
                    raise Exception("The XML file of the video ID %s is not valid. Duplicated object ID in frame %s" %(k,frame.attrib['ID']))
                frameObjectsDict[obj.attrib['ID']] = {}
                objectsDict[obj.attrib['ID']] = {}
 
        #rrc_evaluation_funcs.validate_lines_in_file(k,subm[k],evaluationParams['CRLF'],True,False)
    

def evaluate_method(gtFilePath, submFilePath, evaluationParams):
    """
    Method evaluate_method: evaluate method and returns the results
        Results. Dictionary with the following values:
        - method (required)  Global method metrics. Ex: { 'Precision':0.8,'Recall':0.9 }
        - samples (optional) Per sample metrics. Ex: {'sample1' : { 'Precision':0.8,'Recall':0.9 } , 'sample2' : { 'Precision':0.8,'Recall':0.9 }
    """  
    
    for module,alias in evaluation_imports().items():
        globals()[alias] = importlib.import_module(module) 
    
    gt = rrc_evaluation_funcs.load_zip_file(gtFilePath,evaluationParams['GT_SAMPLE_NAME_2_ID'])
    subm = rrc_evaluation_funcs.load_zip_file(submFilePath,evaluationParams['DET_SAMPLE_NAME_2_ID'],True)
   
    methodMetrics = ['MOTP','MOTA','MOTAN','IDF1','MT','PT','ML','FP','SW','MS']
    methodMetricsSum = { metric:0 for metric in methodMetrics}

    perSampleMetrics = {}
    
    for resFile in tqdm(gt):
        print(resFile)
        sampleResults = { 'frames': {}, 'gt':{} , 'dt':{}}
        
        gtRoot = ET.fromstring(gt[resFile])
        
        # evaluate the GT comparing the GT and Detection XMLs 

        # First build two lists of dictionaries with IDs as keys and bbox coords as values (one dict per frame)
        gtBoxes  = {}
        detBoxes = {}
        dontCareFrameCoords = {}
        num_dont_care_gt = 0
        num_dont_care_det = 0
        
        #detected objects that are not don't care
        num_detected = 0        
        
        
        for gtFrame in list(gtRoot.iter('frame')):

            frameId = gtFrame.attrib['ID']
            
            sampleResults['frames'][frameId] = { "gt":{}, "dt":{} }

            gt_frame_dict = {}
            
            
            for gtObj in list(gtFrame.iter('object')):
                coords = []
                for gtPoint in list(gtObj.iter('Point')):
                    coords.append(max(0,int(gtPoint.attrib['x'])))
                    coords.append(max(0,int(gtPoint.attrib['y'])))
                    
                sampleResults['frames'][frameId]['gt'][str(gtObj.attrib['ID'])] = {"p":coords,"r":"","detId":""};

                if ('#' in gtObj.attrib['Transcription']):
#                 if ('#' in gtObj.attrib['Transcription']) or (gtObj.attrib['ID'] not in gtObjectsDictTxt):
                    if frameId not in dontCareFrameCoords:
                        dontCareFrameCoords[frameId] = []
                    dontCareFrameCoords[frameId].append(coords)
                    
                    num_dont_care_gt = num_dont_care_gt + 1
                    
                    if not gtObj.attrib['ID'] in sampleResults['gt']:
                        sampleResults['gt'][gtObj.attrib['ID']] = {frameId:"DC"};
                    else:
                        sampleResults['gt'][gtObj.attrib['ID']][frameId] = "DC";
                        
                    sampleResults['frames'][frameId]['gt'][gtObj.attrib['ID']]['r'] = "DC";
                    
                else:
                    gt_frame_dict[str(gtObj.attrib['ID'])] = coords

            gtBoxes[frameId] = gt_frame_dict

        
        if resFile in subm:
            
            detRoot = ET.fromstring(subm[resFile])

            for detFrame in list(detRoot.iter('frame')):
                
                frameId = detFrame.attrib['ID']

                det_frame_dict = {}
                
                for detObj in list(detFrame.iter('object')):

                    detCoords = []
                    for detPoint in list(detObj.iter('Point')):
                        detCoords.append(max(0,int(detPoint.attrib['x'])))
                        detCoords.append(max(0,int(detPoint.attrib['y'])))
                        
                    sampleResults['frames'][frameId]['dt'][str(detObj.attrib['ID'])] = {"p":detCoords,"r":"","gtId":""};

                    #check if detection overlaps with Ground Truth object, if so, mark as dont care and exclued it from calculation
                    isDontCare = False;
                    if frameId in dontCareFrameCoords:
                        dontCaresGt = dontCareFrameCoords[frameId];
                        for gtCoords in dontCaresGt:
                            overlapping = overlapping_fn( gtCoords, detCoords)
                            if (overlapping>0.5):
                                isDontCare = True;
                                
                                if not detObj.attrib['ID'] in sampleResults['dt']:
                                    sampleResults['dt'][detObj.attrib['ID']] = {frameId:"DC"};
                                else:
                                    sampleResults['dt'][detObj.attrib['ID']][frameId] = "DC";

                                sampleResults['frames'][frameId]['dt'][detObj.attrib['ID']]['r'] = "DC";
                                
                                num_dont_care_det = num_dont_care_det + 1
                                
                    if not isDontCare:
                        det_frame_dict[str(detObj.attrib['ID'])] = detCoords
                        num_detected += 1                        
                        
                detBoxes[frameId] = det_frame_dict
                
        sampleResults['DC_GT'] = num_dont_care_gt
        sampleResults['DC_DT'] = num_dont_care_det
        
        #Motmetrics fails if no detection on the whole image. Skip the calculation
        if(num_detected==0):
            sampleResults['MOTA'] = 0
            sampleResults['MOTAN'] = 0
            sampleResults['MOTP'] = 0
            sampleResults['IDF1'] = 0
            sampleResults['IDF1'] = 0
            sampleResults['DE'] = 0
            sampleResults['MT'] = 0
            sampleResults['PT'] = 0
            sampleResults['ML'] = 0
        else:
            acc = mm.MOTAccumulator(auto_id=False)
            for frameId in gtBoxes.keys():
                distances_arr = []
                for _, gtCoordsArray in gtBoxes[frameId].items():
                    gtObjectDistances = []
                    if frameId in detBoxes:
                        for _, detCoordsArray in detBoxes[frameId].items():
                            distance = distance_fn( gtCoordsArray, detCoordsArray )
                            gtObjectDistances.append( 1-distance if distance>0.5 else np.nan )
                    distances_arr.append(gtObjectDistances)
                acc.update(
                    list(gtBoxes[frameId].keys()),                                     # Ground truth objects in this frame
                    list(detBoxes[frameId].keys() if frameId in detBoxes else []),  # Detector hypotheses in this frame
                    distances_arr,
                    int(frameId)
                )
            mh = mm.metrics.create()

            summary_vars = ['num_frames', 'mota', 'num_matches','num_switches','num_false_positives','num_objects','num_predictions','num_misses','num_detections','mostly_tracked','partially_tracked','mostly_lost','num_unique_objects','idf1']
            if resFile in subm:
                summary_vars.append('motp')

            summary = mh.compute(acc, metrics=summary_vars, name='acc')

            variables= {'mota':'MOTA', 'num_matches':'MA','num_switches':'SW','num_false_positives':'FP','num_misses':'MS','num_detections':'DE','num_objects':'OB','num_predictions':'PR','mostly_tracked':'MT','partially_tracked':'PT','mostly_lost':'ML','num_unique_objects':'UO','idf1':'IDF1'}

            for variable,alias in variables.items():
                sampleResults[alias] = convert_for_json(summary[variable].acc)

            if resFile in subm:
                sampleResults['motp'] = 0 if np.isnan(summary['motp'].acc) else convert_for_json(summary['motp'].acc)
                sampleResults['MOTAN'] = 0 if sampleResults['PR']==0 or sampleResults['OB']==0 else 0.5 * (sampleResults['FP'] + sampleResults['SW'])/ sampleResults['PR'] + 0.5 * (sampleResults['MS'])/ (sampleResults['OB'] )
            else:
                sampleResults['motp'] = 0
                sampleResults['MOTAN'] = 0

            #motp as MOTchallenge
            sampleResults['MOTP'] = 0 if sampleResults['DE']==0 else 1-float(sampleResults['motp'])




            # get bb information
            df = acc.mot_events
            for frameId,newDF in df.groupby(level=0):

                for eventId,_ in newDF.groupby(level=1):
                    frameResult = sampleResults['frames'][str(frameId)]
                    data = df.loc[frameId,eventId]
                    
                    if (data.OId is not np.nan) and (not math.isnan(data.OId)):
#                     if (data.OId is not np.nan):
                        gtId = str(int(data.OId))
                        frameResult['gt'][gtId]['r'] = data.Type

                        if not gtId in sampleResults['gt']:
                            sampleResults['gt'][gtId] = {frameId:data.Type};
                        else:
                            sampleResults['gt'][gtId][frameId] = data.Type;                 

                        if (data.HId is not np.nan) and (not math.isnan(data.HId)):
                            frameResult['gt'][gtId]['dtId'] = convert_for_json(data.HId)

                    if (data.HId is not np.nan) and (not math.isnan(data.HId)):
                        dtId = str(int(data.HId))
                        frameResult['dt'][dtId]['r'] = data.Type

                        if not dtId in sampleResults['dt']:
                            sampleResults['dt'][dtId] = {frameId:data.Type};
                        else:
                            sampleResults['dt'][dtId][frameId] = data.Type;                    

                        if (data.OId is not np.nan) and (not math.isnan(data.OId)):
                            frameResult['dt'][dtId]['gtId'] = convert_for_json(data.OId)      
        print("num_false_positives:",sampleResults['FP'])
        print("num_switches:",sampleResults['SW'])
        print("num_false_negatives:",sampleResults['MS'])
        #acumulate sampleResults for the method
        methodMetricsSum = { metric: methodMetricsSum[metric] +  sampleResults[metric] for metric in methodMetrics}

        perSampleMetrics[resFile] = sampleResults
    
    print("final num_false_positives:",methodMetricsSum['FP'])
    print("final num_switches:",methodMetricsSum['SW'])
    print("final num_false_negatives:",methodMetricsSum['MS'])
    methodMetrics = {
                    'MOTP': methodMetricsSum['MOTP'] / len(gt),
                    'MOTA':methodMetricsSum['MOTA'] / len(gt),
                    'IDF1':methodMetricsSum['IDF1'] / len(gt),
                    'MOTAN':methodMetricsSum['MOTAN'] / len(gt),
                    'MT':methodMetricsSum['MT'],
                    'PT':methodMetricsSum['PT'],
                    'ML':methodMetricsSum['ML']
                    }

    resDict = {'method': methodMetrics,'per_sample': perSampleMetrics}#,'output_items':output_items
    return resDict;

def convert_for_json(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def polygon_from_points(points):
    """
    Returns a Polygon object to use with the Polygon2 class from a list of 8 points: x1,y1,x2,y2,x3,y3,x4,y4
    """        
    resBoxes=np.empty([1,8],dtype='int32')
    resBoxes[0,0]=int(points[0])
    resBoxes[0,4]=int(points[1])
    resBoxes[0,1]=int(points[2])
    resBoxes[0,5]=int(points[3])
    resBoxes[0,2]=int(points[4])
    resBoxes[0,6]=int(points[5])
    resBoxes[0,3]=int(points[6])
    resBoxes[0,7]=int(points[7])
    pointMat = resBoxes[0].reshape([2,4]).T
    return plg.Polygon( pointMat) 
        
def get_union(pD,pG):
    areaA = pD.area();
    areaB = pG.area();
    return 0 if areaA == 0 or areaB == 0 else areaA + areaB - get_intersection(pD, pG);
    
def get_intersection_over_union(pD,pG):
    union = get_union(pD, pG);
    return 0 if union == 0 else get_intersection(pD, pG) / union
    
def get_intersection(pD,pG):
    if(pD.area()==0):
        return 0
    if(pG.area()==0):
        return 0
    pInt = pD & pG
    if len(pInt) == 0:
        return 0
    return pInt.area()

    
def overlapping_fn(bboxesGT, bboxesDET ):

     pD = polygon_from_points(bboxesDET)
     pD = plg.Utils.convexHull(pD)
        
     pG = polygon_from_points(bboxesGT)
     pG = plg.Utils.convexHull(pG)
     
     intersect = get_intersection(pD,pG) / pD.area();
     
     return intersect


def distance_fn(bboxesGT, bboxesDET ):

     pD = polygon_from_points(bboxesDET)
     pD = plg.Utils.convexHull(pD)
        
     pG = polygon_from_points(bboxesGT)
     pG = plg.Utils.convexHull(pG)
     
     iou = get_intersection_over_union(pD,pG)

     return iou

if __name__=='__main__':
    params = {}
    params['g'] = "./GT_sports.zip"
    params['s'] = "./preds.zip"
    rrc_evaluation_funcs.main_evaluation(params,default_evaluation_params,validate_data,evaluate_method)

