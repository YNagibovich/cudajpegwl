/*
	u-pacs DICOM library
	Author : Y.Nagibovich
*/

#include "dcm_version.h"
#include "dcm_wsi.h"
#include "dcm_file.h"
#include "dcm_utils.h"
#include <string.h>
#include <stdio.h>
#include <direct.h>

// JPEG LS support
//#include ".\ls\jpeg.h"
//#include ".\ls\mcu.h"
//#include ".\ls\proto.h"

int CDICOMFile::CalcSeqLen(dcm_sequence_t& info, int nStart)
{
	int retval = 0, i = nStart; // skip first item SQ tag

	for (i; i < info.tags.size(); i++)
	{
		if (info.tags[i].size != -1)
			retval += info.tags[i].size;
		retval += 8; // TAG+VR+LEN
		if (IsBigData(info.tags[i].vr))
			retval += 4;
	}
	return retval;
}

// used for list of sequences
int CDICOMFile::CalcSeqLen(dcm_sequence_t& info)
{
	int retval = 0, i = 0;

	for (i; i < info.tags.size(); i++)
	{
		if (info.tags[i].size != -1)
			retval += info.tags[i].size;
		retval += 8; // TAG+VR+LEN
		//if (IsBigData(info.tags[i].vr))
		//	retval += 4;
	}
	return retval;
}

int CDICOMFile::SaveTag(int tag, int8_t* dst, tag_list& _items, bool bAsIDX)
{
	int bytesdone = -1;
	uint16_t len;
	int idx, xlen;
	int8_t* ptr = dst;

	if (dst == NULL) 
		return bytesdone;
	if (bAsIDX)
	{
		idx = tag;
		tag = _items[idx].tag.tag;
	}
	else
		idx = FindItemIdx(tag, _items);
	if (idx < 0) 
		return bytesdone;
	// write tag
	//fwrite(&m_items[idx].tag.val[1], sizeof(UINT16), 1, outfile);
	xlen = sizeof(uint16_t);
	memcpy(ptr, &_items[idx].tag.val[1], xlen);
	bytesdone = xlen;
	ptr += xlen;
	//fwrite(&m_items[idx].tag.val[0], sizeof(UINT16), 1, outfile);
	memcpy(ptr, &_items[idx].tag.val[0], xlen);
	bytesdone += xlen;
	ptr += xlen;
	// write VR
	if (tag != 0xFFFEE000 && tag != 0xFFFEE0DD)
	{
		//fwrite(&m_items[idx].vr, sizeof(UINT16), 1, outfile);
		memcpy(ptr, &_items[idx].vr, xlen);
		bytesdone += xlen;
		ptr += xlen;
	}
	// write len
	if (IsBigData(_items[idx].vr))
	{
		len = 0;
		if (tag != 0xFFFEE000 && tag != 0xFFFEE0DD)
		{
			//fwrite(&len, sizeof(UINT16), 1, outfile);
			memcpy(ptr, &len, xlen);
			bytesdone += xlen;
			ptr += xlen;
		}
		//fwrite(&m_items[idx].size, sizeof(UINT32), 1, outfile);
		xlen = sizeof(uint32_t);
		memcpy(ptr, &_items[idx].size, xlen);
		bytesdone += xlen;
		ptr += xlen;
	}
	else
	{
		len = (uint16_t)_items[idx].size;
		//fwrite(&len, sizeof(UINT16), 1, outfile);
		memcpy(ptr, &len, xlen);
		bytesdone += xlen;
		ptr += xlen;

	}
	// write data
	//if (tag != 0xFFFEE000 &&_items[idx].data_ptr && _items[idx].size != -1)
	if (_items[idx].data_ptr && _items[idx].size>0)
	{
		//fwrite(m_items[idx].data_ptr, m_items[idx].size, 1, outfile);
		xlen = _items[idx].size;
		memcpy(ptr, _items[idx].data_ptr, xlen);
		bytesdone += xlen;
		//ptr += xlen;
	}
	return bytesdone;
}

bool CDICOMFile::CreateDICOMFile(dcm_creation_t* ctx, bool bIsWSI)
{
	bool retval=false;

	//ctx->image_info.SetImage(image, ctx->image_info.image_type, ctx->image_info.image_quality); 
	ctx->m_bWSImode = bIsWSI;
	if (bIsWSI)
		ctx->SetWSIMode();
	// save meta info
	retval = AddDCMItem( 0x00020001, &ctx->meta_info.FileMetaInformationVersion, m_items);
	if (retval) retval = AddDCMItem(0x00020002, ctx->meta_info.MediaStorageSOPClassUID.c_str(), m_items);
	if (retval) retval = AddDCMItem(0x00020003, ctx->meta_info.MediaStorageSOPInstanceUID.c_str(), m_items);
	if (retval) retval = AddDCMItem(0x00020010, ctx->meta_info.TransferSyntaxUID.c_str(), m_items);
	if (retval) retval = AddDCMItem(0x00020012, ctx->meta_info.ImplementationClassUID.c_str(), m_items);
	if (retval) retval = AddDCMItem(0x00020013, ctx->meta_info.ImplementationVersionName.c_str(), m_items);
	if (retval) retval = AddDCMItem(0x00020016, ctx->meta_info.SourceApplicationEntityTitle.c_str(), m_items);
	// save modality info
	if (retval) retval = AddDCMItem(0x00080005, ctx->modality_info.SpecificCharacterSet.c_str(), m_items);
	if (retval) retval = AddDCMItem(0x00080008, ctx->modality_info.ImageType.c_str(), m_items);
	if (retval) retval = AddDCMItem(0x00080016, ctx->modality_info.SOPClassUID.c_str(), m_items);
	if (retval) retval = AddDCMItem(0x00080018, ctx->modality_info.SOPInstanceUID.c_str(), m_items);
	if (retval) retval = AddDCMItem(0x00080020, ctx->modality_info.StudyDate.c_str(), m_items);
	if (retval) retval = AddDCMItem(0x00080021, ctx->modality_info.SeriesDate.c_str(), m_items);
	if (retval) retval = AddDCMItem(0x00080022, ctx->modality_info.AcquisitionDate.c_str(), m_items);
	if (retval) retval = AddDCMItem(0x00080023, ctx->modality_info.ContentDate.c_str(), m_items);
	if (retval) retval = AddDCMItem(0x00080030, ctx->modality_info.StudyTime.c_str(), m_items);
	if (retval) retval = AddDCMItem(0x00080031, ctx->modality_info.SeriesTime.c_str(), m_items);
	if (retval) retval = AddDCMItem(0x00080032, ctx->modality_info.AcquisitionTime.c_str(), m_items);
	if (retval) retval = AddDCMItem(0x00080033, ctx->modality_info.ContentTime.c_str(), m_items);
	if (retval) retval = AddDCMItem(0x00080050, ctx->modality_info.AccessionNumber.c_str(), m_items);
	if (retval) retval = AddDCMItem(0x00080060, ctx->modality_info.Modality.c_str(), m_items);
	if (retval) retval = AddDCMItem(0x00080064, ctx->modality_info.ConversionType.c_str(), m_items);
	if (retval) retval = AddDCMItem(0x00080070, ctx->modality_info.Manufacturer.c_str(), m_items);
	if (retval) retval = AddDCMItem(0x00080080, ctx->modality_info.InstitutionName.c_str(), m_items);
	if (retval) retval = AddDCMItem(0x00080081, ctx->modality_info.InstitutionAddress.c_str(), m_items);
	if (retval) retval = AddDCMItem(0x00080090, ctx->modality_info.ReferringPhysiciansName.c_str(), m_items);
	if (retval) retval = AddDCMItem(0x00081010, ctx->modality_info.StationName.c_str(), m_items);
	if (retval) retval = AddDCMItem(0x00081030, ctx->modality_info.StudyDescription.c_str(), m_items);
	if (retval) retval = AddDCMItem(0x0008103e, ctx->modality_info.SeriesDescription.c_str(), m_items);
	if (retval) retval = AddDCMItem(0x00081050, ctx->modality_info.PerformingPhysiciansName.c_str(), m_items);
	if (retval) retval = AddDCMItem(0x00081070, ctx->modality_info.OperatorsName.c_str(), m_items);
	if (retval) retval = AddDCMItem(0x00081090, ctx->modality_info.ManufacturersModelName.c_str(), m_items);
	if (bIsWSI)
	{
		if (retval) retval = AddDCMItem(0x00089206, ctx->modality_info.VolumetricProperies.c_str(), m_items);
	}
	// save  patient info
	if (retval) retval = AddDCMItem(0x00100010, ctx->patient_info.PatientsName.c_str(), m_items);
	if (retval) retval = AddDCMItem(0x00100020, ctx->patient_info.PatientID.c_str(), m_items);
	if (retval) retval = AddDCMItem(0x00100030, ctx->patient_info.PatientsBirthDate.c_str(), m_items);
	if (retval) retval = AddDCMItem(0x00100040, ctx->patient_info.PatientsSex.c_str(), m_items);
	if (retval) retval = AddDCMItem(0x00101010, ctx->patient_info.PatientsAge.c_str(), m_items);
	if (retval) retval = AddDCMItem(0x00104000, ctx->patient_info.PatientComments.c_str(), m_items);
	// save body part info
	if (retval) retval = AddDCMItem(0x00180015, ctx->bodypart_info.BodyPartExamined.c_str(), m_items);
	if (retval) retval = AddDCMItem(0x00181020, ctx->bodypart_info.SoftwareVersions.c_str(), m_items);
	if (retval) retval = AddDCMItem(0x00181030, ctx->bodypart_info.ProtocolName.c_str(), m_items);
	// save exam info
	if (retval) retval = AddDCMItem(0x0020000d, ctx->exam_info.StudyInstanceUID.c_str(), m_items);
	if (retval) retval = AddDCMItem(0x0020000e, ctx->exam_info.SeriesInstanceUID.c_str(), m_items);
	if (retval) retval = AddDCMItem(0x00200010, ctx->exam_info.StudyID.c_str(), m_items);
	if (retval) retval = AddDCMItem(0x00200011, ctx->exam_info.SeriesNumber.c_str(), m_items);
	if (retval) retval = AddDCMItem(0x00200012, ctx->exam_info.AcquisitionNumber.c_str(), m_items);
	if (retval) retval = AddDCMItem(0x00200013, ctx->exam_info.InstanceNumber.c_str(), m_items);
	if (retval) retval = AddDCMItem(0x00200020, ctx->exam_info.PatientOrientation.c_str(), m_items);
	if (bIsWSI)
	{
		if (retval) retval = AddDCMItem(0x00200052, ctx->exam_info.FrameofReferenceUID.c_str(), m_items); //(0020,0052) UI 
		if (retval) retval = AddDCMItem(0x00201040, ctx->exam_info.PositionReferenceIndicator.c_str(), m_items); //(0020,1040) LO
	}
	// save image info
	if (retval) retval = AddDCMItem(0x00280002, &ctx->image_info.SamplesPerPixel, m_items);
	if (retval) retval = AddDCMItem(0x00280004, ctx->image_info.PhotometricInterpretation.c_str(), m_items);
	if (retval) retval = AddDCMItem(0x00280006, &ctx->image_info.PlanarConfiguration, m_items);
	if (retval) retval = AddDCMItem(0x00280008, ctx->image_info.NumberOfFrames.c_str(), m_items);
	if (retval) retval = AddDCMItem(0x00280010, &ctx->image_info.Rows, m_items);
	if (retval) retval = AddDCMItem(0x00280011, &ctx->image_info.Columns, m_items);
	if (retval) retval = AddDCMItem(0x00280030, ctx->image_info.PixelSpacing.c_str(), m_items);
	if (retval) retval = AddDCMItem(0x00280100, &ctx->image_info.BitsAllocated, m_items);
	if (retval) retval = AddDCMItem(0x00280101, &ctx->image_info.BitsStored, m_items);
	if (retval) retval = AddDCMItem(0x00280102, &ctx->image_info.HighBit, m_items);
	if (retval) retval = AddDCMItem(0x00280103, &ctx->image_info.PixelRepresentation, m_items);
	if(0)//for b&w
	{
		if (retval) retval = AddDCMItem(0x00281050, ctx->image_info.WindowCenter.c_str(), m_items);
		if (retval) retval = AddDCMItem(0x00281051, ctx->image_info.WindowWidth.c_str(), m_items);
		if (retval) retval = AddDCMItem(0x00281052, ctx->image_info.RescaleIntercept.c_str(), m_items);
		if (retval) retval = AddDCMItem(0x00281053, ctx->image_info.RescaleSlope.c_str(), m_items);
		if (retval) retval = AddDCMItem(0x00281055, ctx->image_info.WindowCenterWidthExplanation.c_str(), m_items);
	}
	if(ctx->image_info.image_type!=DCM_JPEG_NONE)
	{
		if (retval) retval = AddDCMItem(0x00282110, ctx->image_info.LossyImageCompression.c_str(), m_items);
		if (retval) retval = AddDCMItem(0x00282112, ctx->image_info.LossyImageCompressionRatio.c_str(), m_items);
		if (retval) retval = AddDCMItem(0x00282114, ctx->image_info.LossyImageCompressionMethod.c_str(), m_items);
	}
	if (bIsWSI)
	{
		dcm_tile_info_t item;
		item.data_ptr = NULL;
		item.size = 0;
		item.x = 0;
		item.y = 0;
		ctx->image_info.tiles.push_back(item);
	}
	// save study info
	if (retval) retval = AddDCMItem(0x00321060, ctx->study_info.RequestedProcedureDescription.c_str(), m_items);
	if (retval) retval = AddDCMItem(0x00324000, ctx->study_info.StudyComments.c_str(), m_items);
	// check WSI
	if (bIsWSI)
	{

		if (retval) retval = AddDCMItem(0x00400512, ctx->wsi_info.ContainerIdentifier.c_str(), m_items); // (0040, 0512) LO 1
		if (retval) retval = AddDCMItem(0x00400513, ctx->wsi_info.IssuerOfTheContainerIdentifierSequence, 0, m_items);// (0040, 0513) SQ  1
		if (retval) retval = AddDCMItem(0x00400515, ctx->wsi_info.AlternateContainerIdentifierSequence, 0, m_items);// (0040, 0515) SQ  1
		if (retval) retval = AddDCMItem(0x00400518, ctx->wsi_info.ContainerTypeCodeSequence, 0, m_items);//(0040, 0518) SQ
		if (retval) retval = AddDCMItem(0x00400555, ctx->wsi_info.AcquisitionContextSequence, 0, m_items); //(0040, 0555) SQ
		// create specimen seq info
		dcm_sequence_t seq;
		seq.tags.clear();
		AddDCMItem(0xFFFEE000, NULL, 0, seq.tags); // first item in sequence
		AddDCMItem(0x00400551, ctx->wsi_info.SpecimenIdentifier.c_str(), seq.tags);
		AddDCMItem(0x00400554, ctx->wsi_info.SpecimenUID.c_str(), seq.tags);
		AddDCMItem(0x00400562, NULL, 0, seq.tags);
		AddDCMItem(0x00400610, NULL, 0, seq.tags);
		if (retval && CreateSequence(seq))
		{
			ctx->wsi_info.SpecimenContextSequence = seq.seq_data;
			retval = AddDCMItem(0x00400560, ctx->wsi_info.SpecimenContextSequence, seq.seq_size, m_items);// (0040, 0560) SQ
		}
		if (retval) retval = AddDCMItem(0x00480001, &ctx->wsi_info.ImagedVolumeWidth, m_items); // (0048, 0001) FL 1
		if (retval) retval = AddDCMItem(0x00480002, &ctx->wsi_info.ImagedVolumeHeight, m_items); // (0048, 0002) FL 1
		if (retval) retval = AddDCMItem(0x00480003, &ctx->wsi_info.ImagedVolumeDepth, m_items); //  (0048, 0003) FL 1
		if (retval) retval = AddDCMItem(0x00480006, &ctx->wsi_info.TotalPixelMatrixColumns, m_items); // (0048, 0006) UL 1
		if (retval) retval = AddDCMItem(0x00480007, &ctx->wsi_info.TotalPixelMatrixRows, m_items); // (0048, 0007) UL 1
		// create plane seq info
		seq.tags.clear();
		AddDCMItem(0xFFFEE000, NULL, 0, seq.tags); // first item in sequence
		AddDCMItem(0x0040072A, ctx->wsi_info.XOffsetinSlideCoordinateSystem.c_str(), seq.tags);
		AddDCMItem(0x0040073A, ctx->wsi_info.YOffsetinSlideCoordinateSystem.c_str(), seq.tags);
		AddDCMItem(0x0040074A, ctx->wsi_info.ZOffsetinSlideCoordinateSystem.c_str(), seq.tags);
		if (retval && CreateSequence(seq))
		{
			ctx->wsi_info.TotalPixelMatrixOriginSequence = seq.seq_data;
			retval = AddDCMItem(0x00480008, ctx->wsi_info.TotalPixelMatrixOriginSequence, seq.seq_size, m_items); // (0048, 0008) SQ 1
		}
		
		if (retval) retval = AddDCMItem(0x00480010, ctx->wsi_info.SpecimenLabelInImage.c_str(), m_items); // (0048, 0010) CS 1
		if (retval) retval = AddDCMItem(0x00480011, ctx->wsi_info.FocusMethod.c_str(), m_items); // (0048, 0011) CS  1
		if (retval) retval = AddDCMItem(0x00480012, ctx->wsi_info.ExtendedDepthOfField.c_str(), m_items); // (0048, 0012) CS 1
		if (retval) retval = AddDCMItem(0x00480013, &ctx->wsi_info.NumberOfFocalPlanes, m_items); //  (0048, 0013) US 1
		if (retval) retval = AddDCMItem(0x00480014, &ctx->wsi_info.DistanceBetweenFocalPlanes, m_items); // (0048, 0014) FL 1
		if (retval) retval = AddDCMItem(0x00480015, &ctx->wsi_info.RecommendedAbsentPixelCIELabValue, m_items); // (0048, 0015) US 3
		if (retval) retval = AddDCMItem(0x00480100, ctx->wsi_info.IlluminatorTypeCodeSequence, 0, m_items); // (0048, 0100) SQ 1
		if (retval) retval = AddDCMItem(0x00480102, ctx->wsi_info.ImageOrientationSlide.c_str(), m_items); // (0048, 0102) DS 6
		if (retval) retval = AddDCMItem(0x00480105, ctx->wsi_info.OpticalPathSequence, 0, m_items); // (0048, 0105) SQ 1
		if (retval) retval = AddDCMItem(0x52009229, ctx->wsi_info.SharedFunctionalGroupsSequence, 0, m_items);  // (5200, 9229) SQ     1
		// create tiles seq info
		if (retval) retval = AddDCMItem(0x52009230, ctx->wsi_info.PerFrameFunctionalGroupsSequence, 0, m_items); // (5200, 9230) SQ                 1
	}
	//save image data
	if (bIsWSI)
	{
		if (retval) retval = AddDCMItem(0x7fe00010, NULL, -1, m_items);						// OB 
		// add sequence item 0xFFFEE000 with 0 length 
	}
	else
		if (retval) retval = AddDCMItem(0x7fe00010, ctx->image_info.PixelData, ctx->image_info.image_length, m_items);						// OW 
	//DataSetTrailingPadding;				//(0xfffcfffc) OB 
	return retval;
}

// TBD memory leaks !!!

bool CDICOMFile::AddTiles(dcm_tiles_list* info)
{
	uint32_t i, perframe_idx = -1;
	std::string cpos = "0";

	dcm_sequence_t seq;
	dcm_sequence_t frame_content_seq;
	dcm_sequence_t plane_pos_seq;
	
	dcm_sequence_t perframe_seq;

	perframe_seq.tags.clear();
	perframe_idx = FindItemIdx(0x52009230, m_items);
	for (i = 0; i < info->size(); i++)
	{
		if (perframe_idx >0 && info->at(i).size != 0)
		{
			// frame content seq
			frame_content_seq.tags.clear();
			AddDCMItem(0xFFFEE000, NULL, 0, frame_content_seq.tags); // first item in sequence
			AddDCMItem(0x00209157, &i, frame_content_seq.tags);
			CreateSequence(frame_content_seq);
			
			// plane pos seq
			plane_pos_seq.tags.clear();
			AddDCMItem(0xFFFEE000, NULL, 0, plane_pos_seq.tags); // first item in sequence
			AddDCMItem(0x0040072A, cpos.c_str(), plane_pos_seq.tags); // X
			AddDCMItem(0x0040073A, cpos.c_str(), plane_pos_seq.tags); // Y
			AddDCMItem(0x0040074A, cpos.c_str(), plane_pos_seq.tags); // Z in mm
			AddDCMItem(0x0048021E, &info->at(i).x, plane_pos_seq.tags); // X in px start from 0
			AddDCMItem(0x0048021F, &info->at(i).y, plane_pos_seq.tags); // Y in px start from 0
			CreateSequence(plane_pos_seq);

			// create perframe seq
			seq.tags.clear();
			AddDCMItem(0xFFFEE000, NULL, 0, seq.tags); // first item in sequence
			AddDCMItem(0x00209111, frame_content_seq.seq_data, frame_content_seq.seq_size, seq.tags);
			AddDCMItem(0x0048021A, plane_pos_seq.seq_data, plane_pos_seq.seq_size, seq.tags);
			if (CreateSequence(seq, false))
			{
				AddDCMItem(0xFFFEE000, seq.seq_data, seq.seq_size, perframe_seq.tags);
			}
		}
		// add tile data
		
		AddDCMItem(0xFFFEE000, info->at(i).data_ptr, info->at(i).size, m_items);
	}
	// update sequences
	if (perframe_idx > 0)
	{
		if (CreateSequence(perframe_seq, true))
		{
			m_items[perframe_idx].data_ptr = perframe_seq.seq_data;
			m_items[perframe_idx].size = perframe_seq.seq_size;
		}
	}
	// add ending tag 0xFFFEE0DD
	AddDCMItem(0xFFFEE0DD, NULL, 0, m_items);
	return true;
}

bool CDICOMFile::CreateSequence(dcm_sequence_t& info, bool bAlt)
{
	int8_t* ptr;
	int offset = 0, i = 0;

	if (info.tags.size() == 0)
		return false;
	if (!bAlt)
	{
		info.tags[0].size = CalcSeqLen(info, 1); // do not calc 1st item
		info.seq_size = info.tags[0].size + 8;
	}
	else
		info.seq_size = CalcSeqLen(info);
	info.seq_data = (int8_t*)malloc(info.seq_size);
	// copy data
	ptr = info.seq_data;
	for (i = 0; i < info.tags.size(); i++)
	{
		if ( bAlt)
			offset = SaveTag(i, ptr, info.tags, true);
		else
			offset = SaveTag(info.tags[i].tag.tag, ptr, info.tags);
		if ( offset<0)
			continue;
		ptr += offset;
	}
	return true;
}