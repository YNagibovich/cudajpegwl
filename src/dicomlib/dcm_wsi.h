/*
	u-pacs DICOM library
	Author : Y.Nagibovich
*/

#pragma once
#include "dcm_version.h"
#include "dcm_data.h"
#include "dcm_constants.h"

#include <string>
#include <vector>
#include <queue>

typedef struct
{
	int8_t* data_ptr;		// pointer to data
	int size;
	int x;
	int y;
} dcm_tile_info_t;

typedef std::vector<dcm_tile_info_t> dcm_tiles_list;

typedef struct
{
	// container info
	std::string ContainerIdentifier; // (0040, 0512) LO 1
	void* IssuerOfTheContainerIdentifierSequence; // (0040, 0513) SQ  1
	void* AlternateContainerIdentifierSequence; // (0040, 0515) SQ  1
	void* ContainerTypeCodeSequence; //(0040, 0518) SQ
	void* AcquisitionContextSequence; //(0040, 0555) SQ
	void* SpecimenContextSequence; //(0040, 0560) SQ
	
	/*
	
	// plane position
	Image Center Point Coordinates Sequence
	(0040, 071A)
	2
	The coordinates of the center point of the Image in the Slide Coordinate System Frame of Reference.Zero or one item shall be present in the sequence.See Section C.8.12.2.1.1 for further explanation.
	*/
	std::string XOffsetinSlideCoordinateSystem; //(0040, 072A) DS 1 in mm
	std::string YOffsetinSlideCoordinateSystem; //(0040, 073A) DS 1 in mm
	std::string ZOffsetinSlideCoordinateSystem; //(0040, 074A) DS 1 in mm
	
	std::string SpecimenIdentifier; //	0x00400551, VR_LO, "",
	std::string SpecimenUID; //	0x00400554, VR_UI, "",
	// whole image info
	float ImagedVolumeWidth; // (0048, 0001) FL 1
	float ImagedVolumeHeight; // (0048, 0002) FL 1
	float ImagedVolumeDepth; //  (0048, 0003) FL 1
	  
	uint32_t TotalPixelMatrixColumns; // (0048, 0006) UL 1
	uint32_t TotalPixelMatrixRows; // (0048, 0007) UL 1

	void* TotalPixelMatrixOriginSequence; // (0048, 0008) SQ 1
	std::string SpecimenLabelInImage; // (0048, 0010) CS 1
	std::string FocusMethod; // (0048, 0011) CS  1
	std::string ExtendedDepthOfField; // (0048, 0012) CS 1
	uint16_t NumberOfFocalPlanes; //  (0048, 0013) US 1
	float DistanceBetweenFocalPlanes; // (0048, 0014) FL 1
	uint16_t RecommendedAbsentPixelCIELabValue; // (0048, 0015);US 3
	void* IlluminatorTypeCodeSequence; // (0048, 0100) SQ 1
	std::string ImageOrientationSlide; // (0048, 0102) DS 6
	void* OpticalPathSequence; // (0048, 0105) SQ 1

/*
(0048, 0106)
Optical Path Identifier
OpticalPathIdentifier
SH
1
(0048, 0107)
Optical Path Description
OpticalPathDescription
ST
1
(0048, 0108)
Illumination Color Code Sequence
IlluminationColorCodeSequence
SQ
1
(0048, 0110)
Specimen Reference Sequence
SpecimenReferenceSequence
SQ
1
(0048, 0111)
Condenser Lens Power
CondenserLensPower
DS
1
(0048, 0112)
Objective Lens Power
ObjectiveLensPower
DS
1
(0048, 0113)
Objective Lens Numerical Aperture
ObjectiveLensNumericalAperture
DS
1
(0048, 0120)
Palette Color Lookup Table Sequence
PaletteColorLookupTableSequence
SQ
1
(0048, 0200)
Referenced Image Navigation Sequence
ReferencedImageNavigationSequence
SQ
1
(0048, 0201)
Top Left Hand Corner of Localizer Area
TopLeftHandCornerOfLocalizerArea
US
2
(0048, 0202)
Bottom Right Hand Corner of Localizer Area
BottomRightHandCornerOfLocalizerArea
US
2
(0048, 0207)
Optical Path Identification Sequence
OpticalPathIdentificationSequence
SQ
1
(0048, 021A)
Plane Position(Slide) Sequence
PlanePositionSlideSequence
SQ
1
(0048, 021E)
Column Position In Total Image Pixel Matrix
ColumnPositionInTotalImagePixelMatrix
SL
1
(0048, 021F)
Row Position In Total Image Pixel Matrix
RowPositionInTotalImagePixelMatrix
SL
1
(0048, 0301)
Pixel Origin Interpretation
PixelOriginInterpretation
CS
1
*/

	// slice spec
	void* SharedFunctionalGroupsSequence; // (5200, 9229) SQ     1
	// one slice
	// 0020,9110 SQ
	//	0018,0050	DS Slice thickness "1.11"
	//	0028,0030	Pixel spacing "0.000454\0.000454"

	// tile spec
	void* PerFrameFunctionalGroupsSequence; // (5200, 9230) SQ                 1
	// for every tile
	// 0020,9111 SQ
	//		0020,9157 UL	tile idx, start from 1	
	// 0048,021A SQ
	//		0040, 072A	DS
	//		0040, 073A	DS
	//		0040, 074A	DS
	//		0048, 021E	SL x-pos
	//		0048, 021F	SL y-pos

	void Clear();
} dcm_WSI_info_t;
