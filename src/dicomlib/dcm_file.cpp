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

//static int IMA_len=0;
//static int IMA_off=0;
//char* IMA_ptr=0;
//FILE* lsj_outFile=NULL;

//extern int inputBufferOffset;
//extern int numInputBytes;                    /* bytes in inputBuffer       */
//extern int maxInputBytes;                    /* Size of inputBuffer        */
/*
std::string FindExtension(CString& name)
{	
	std::string ret = "";
	int len = name.GetLength();
	int i;

	for(i = len-1; i >= 0; i--)
	{	
		if(name[i] == '.')
		{	
			ret = CW2A(name.Mid(i+1)).m_psz;
			break;
		}
	}
	return ret;
}
*/
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////

#ifdef USE_XIMAGE

int FindType( std::string& ext)
{	
	int type = 0;

	if( ext.size()==0) return CXIMAGE_FORMAT_UNKNOWN;
	if (ext == "bmp")					type = CXIMAGE_FORMAT_BMP;
#if CXIMAGE_SUPPORT_JPG
	else if (ext=="jpg"||ext=="jpeg")	type = CXIMAGE_FORMAT_JPG;
#endif
#if CXIMAGE_SUPPORT_GIF
	else if (ext == "gif")				type = CXIMAGE_FORMAT_GIF;
#endif
#if CXIMAGE_SUPPORT_PNG
	else if (ext == "png")				type = CXIMAGE_FORMAT_PNG;
#endif
#if CXIMAGE_SUPPORT_MNG
	else if (ext=="mng"||ext=="jng")	type = CXIMAGE_FORMAT_MNG;
#endif
#if CXIMAGE_SUPPORT_ICO
	else if (ext == "ico")				type = CXIMAGE_FORMAT_ICO;
#endif
#if CXIMAGE_SUPPORT_TIF
	else if (ext=="tiff"||ext=="tif")	type = CXIMAGE_FORMAT_TIF;
#endif
#if CXIMAGE_SUPPORT_TGA
	else if (ext=="tga")				type = CXIMAGE_FORMAT_TGA;
#endif
#if CXIMAGE_SUPPORT_PCX
	else if (ext=="pcx")				type = CXIMAGE_FORMAT_PCX;
#endif
#if CXIMAGE_SUPPORT_WBMP
	else if (ext=="wbmp")				type = CXIMAGE_FORMAT_WBMP;
#endif
#if CXIMAGE_SUPPORT_WMF
	else if (ext=="wmf"||ext=="emf")	type = CXIMAGE_FORMAT_WMF;
#endif
#if CXIMAGE_SUPPORT_J2K
	else if (ext=="j2k"||ext=="jp2")	type = CXIMAGE_FORMAT_J2K;
#endif
#if CXIMAGE_SUPPORT_JBG
	else if (ext=="jbg")				type = CXIMAGE_FORMAT_JBG;
#endif
#if CXIMAGE_SUPPORT_JP2
	else if (ext=="jp2"||ext=="j2k")	type = CXIMAGE_FORMAT_JP2;
#endif
#if CXIMAGE_SUPPORT_JPC
	else if (ext=="jpc"||ext=="j2c")	type = CXIMAGE_FORMAT_JPC;
#endif
#if CXIMAGE_SUPPORT_PGX
	else if (ext=="pgx")				type = CXIMAGE_FORMAT_PGX;
#endif
#if CXIMAGE_SUPPORT_RAS
	else if (ext=="ras")				type = CXIMAGE_FORMAT_RAS;
#endif
#if CXIMAGE_SUPPORT_PNM
	else if (ext=="pnm"||ext=="pgm"||ext=="ppm") type = CXIMAGE_FORMAT_PNM;
#endif
	else type = CXIMAGE_FORMAT_UNKNOWN;
	return type;
}

bool CDICOMFile::SetWL(WL_t* wlt, CxImage* image)
{
	uint16_t h, w;
	bool retval;

	if (image == NULL) 
		return false;
	GetDCMItem(0x00280010, &h);
	GetDCMItem(0x00280011, &w);
	if (HasWL()) 
		retval = image->CreateFromArray(GetImageWL(wlt), w, h, 8, w, true);
	else 
		retval = image->CreateFromArray(GetImagePtr(), w, h, GetDCMbpp(), w*(GetDCMbpp() / 8), true);
	return retval;
}


bool CDICOMFile::LoadImage(CxImage *image, int image_id)
{
	int z;
	WL_t WL;
	bool image_WL;
	int16_t w = 0, h = 0;
	bool retval = false;
	int8_t* image_ptr;
	int image_length;
	int bpp, lossless = 0;
	char* new_image_ptr = NULL;
	int new_image_size = 0;
	//char temp_path[MAX_PATH];
	char temp_file[MAX_PATH];
	DecompressInfo dcInfo;

	z = GetDCMCompression(&lossless);
	image_ptr = GetImagePtr(image_id);
	image_length = GetImageLength(image_id);
	if (image_ptr == NULL) return retval;
	image_WL = GetWL(&WL);
	bpp = GetDCMbpp();
	GetDCMItem(0x00280010, &h);
	GetDCMItem(0x00280011, &w);
	if (h == 0) return retval;
	if (w == 0) return retval;
	if (z == 0)
	{
		if (image_WL) retval = image->CreateFromArray(GetImageWL(&WL, image_id), w, h, 8, w, true);
		else retval = image->CreateFromArray(image_ptr, w, h, bpp, w*(bpp / 8), true);
	}
	else
	{
		if (z > 0)
		{
			if (lossless)
			{
				memset(&dcInfo, 0, sizeof(dcInfo));
				IMA_len = image_length;
				IMA_off = 0;
				IMA_ptr = (char*)image_ptr;
				lsj_outFile = NULL;

				// TODO !!!!!!!!!!!!!!!!!!!!!!!!!!!! 
				inputBufferOffset = 0;
				numInputBytes = 0;
				maxInputBytes = 0;
				if (!ReadFileHeader(&dcInfo)) retval = false;
				else if (!ReadScanHeader(&dcInfo)) retval = false;
				else
				{
					//GetTempPath( MAX_PATH, temp_path);
					//GetTempFileName( temp_path, "UPACS", 0, temp_file);
					strcpy(temp_file, GetTempFile().c_str());
					lsj_outFile = fopen(temp_file, "w+b");
					if (lsj_outFile)
					{
						new_image_size = w * h * bpp / 8;
						new_image_ptr = (char*)malloc(new_image_size);
						if (new_image_ptr)
						{
							DecoderStructInit(&dcInfo);
							HuffDecoderInit(&dcInfo);
							DecodeImage(&dcInfo, new_image_ptr);
							FreeArray2D((char**)mcuROW1);
							FreeArray2D((char**)mcuROW2);
							fseek(lsj_outFile, 0, SEEK_SET);
							new_image_size = fread(new_image_ptr, 1, new_image_size, lsj_outFile);
							retval = image->CreateFromArray((uint8_t*)new_image_ptr, w, h, bpp, w*(bpp / 8), true);
							free(new_image_ptr);
							fclose(lsj_outFile);
							CString s;
							s = temp_file;
							DeleteFile((LPCTSTR)s);
						}
						else retval = false;
					}
					else retval = false;
				}
			}
			else retval = image->Decode(image_ptr, image_length, CXIMAGE_FORMAT_JPG);
		}
		else
		{
			if (image_length < (w*h*(bpp / 8))) retval = image->Decode(image_ptr, image_length, CXIMAGE_FORMAT_JPG);
			else
			{
				if (image_WL) retval = image->CreateFromArray(GetImageWL(&WL, image_id), w, h, 8, w, true);
				else retval = image->CreateFromArray(image_ptr, w, h, bpp, w*(bpp / 8), true);
			}
		}
	}
	return retval;
}

bool CDICOMFile::CreateDICOMFile(dcm_creation_t* ctx, CxImage* image)
{
	bool retval = false;

	ctx->image_info.SetImage(image, ctx->image_info.image_type, ctx->image_info.image_quality);
	// save meta info
	retval = AddDCMItem(0x00020001, &ctx->meta_info.FileMetaInformationVersion, m_items);
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
	// save image info
	if (retval) retval = AddDCMItem(0x00280002, &ctx->image_info.SamplesPerPixel, m_items);
	if (retval) retval = AddDCMItem(0x00280004, ctx->image_info.PhotometricInterpretation.c_str(), m_items);
	if (retval) retval = AddDCMItem(0x00280006, &ctx->image_info.PlanarConfiguration, m_items);
	if (retval) retval = AddDCMItem(0x00280010, &ctx->image_info.Rows, m_items);
	if (retval) retval = AddDCMItem(0x00280011, &ctx->image_info.Columns, m_items);
	if (retval) retval = AddDCMItem(0x00280030, ctx->image_info.PixelSpacing.c_str(), m_items);
	if (retval) retval = AddDCMItem(0x00280100, &ctx->image_info.BitsAllocated, m_items);
	if (retval) retval = AddDCMItem(0x00280101, &ctx->image_info.BitsStored, m_items);
	if (retval) retval = AddDCMItem(0x00280102, &ctx->image_info.HighBit, m_items);
	if (retval) retval = AddDCMItem(0x00280103, &ctx->image_info.PixelRepresentation, m_items);
	if (0)//for b&w
	{
		if (retval) retval = AddDCMItem(0x00281050, ctx->image_info.WindowCenter.c_str(), m_items);
		if (retval) retval = AddDCMItem(0x00281051, ctx->image_info.WindowWidth.c_str(), m_items);
		if (retval) retval = AddDCMItem(0x00281052, ctx->image_info.RescaleIntercept.c_str(), m_items);
		if (retval) retval = AddDCMItem(0x00281053, ctx->image_info.RescaleSlope.c_str(), m_items);
		if (retval) retval = AddDCMItem(0x00281055, ctx->image_info.WindowCenterWidthExplanation.c_str(), m_items);
	}
	if (ctx->image_info.image_type != DCM_JPEG_NONE)
	{
		if (retval) retval = AddDCMItem(0x00282110, ctx->image_info.LossyImageCompression.c_str(), m_items);
		if (retval) retval = AddDCMItem(0x00282112, ctx->image_info.LossyImageCompressionRatio.c_str(), m_items);
		if (retval) retval = AddDCMItem(0x00282114, ctx->image_info.LossyImageCompressionMethod.c_str(), m_items);
	}
	// save study info
	if (retval) retval = AddDCMItem(0x00321060, ctx->study_info.RequestedProcedureDescription.c_str(), m_items);
	if (retval) retval = AddDCMItem(0x00324000, ctx->study_info.StudyComments.c_str(), m_items);
	//save image data
	if (retval) retval = AddDCMItem(0x7fe00010, ctx->image_info.PixelData, ctx->image_info.image_length, m_items);						// OW 
	//DataSetTrailingPadding;				//(0xfffcfffc) OB 
	return retval;
}

bool CDICOMFile::CreateDICOMFile(dcm_creation_t* ctx, char* src_file, int nQuality)
{
	int type;
	CString s;
	std::string ext;
	s = src_file;

	ext = FindExtension(s);
	type = FindType(ext);

	ctx->image_info.cImage = new CxImage(s, type);
	if (ctx->image_info.cImage == NULL)
		return false;
	if (!ctx->image_info.cImage->IsValid())
	{
		delete ctx->image_info.cImage;
		ctx->image_info.cImage = NULL;
		return false;
	}
	if (type == CXIMAGE_FORMAT_JPG)
		ctx->image_info.image_type = DCM_JPEG_STD;
	ctx->image_info.image_quality = nQuality;
	return CreateDICOMFile(ctx, ctx->image_info.cImage);
}

bool dcm_creation_image_info_t::SetImage(CxImage* image, int jpeg_type, int jpeg_qty)
{
	bool retval = false;
	uint8_t* dptr;
	float ratio = 0;

	if (!image->IsValid())
		return retval;
	retval = true;
	Clear(false);
	if (image->IsGrayScale())
		PhotometricInterpretation = "MONOCHROME2";
	else
		PhotometricInterpretation = "RGB";
	Rows = image->GetHeight();
	Columns = image->GetWidth();
	BitsAllocated = image->GetBpp();
	BitsStored = image->GetBpp();
	image->Flip();
	cImage = image;
	HighBit = BitsStored - 1;
	PixelRepresentation = 0;
	image_type = jpeg_type;
	image_quality = jpeg_qty;
	if (image_type == DCM_JPEG_NONE)
	{
		LossyImageCompression = "00";
		PixelData = (char*)image->GetBits();
		image_length = image->GetSize(); //(BitsAllocated*Rows*Columns);
	}
	else
	{
		LossyImageCompression = "01";
		image->SetJpegQuality((uint8_t)jpeg_qty);
		PixelData = NULL;
		// NOTE !!!!
		// free PixelData manually after usage
		dptr = (uint8_t*)PixelData;
		if (image_type == DCM_JPEG_STD)
		{
			retval = image->Encode(dptr, image_length, CXIMAGE_FORMAT_JPG);
			LossyImageCompressionMethod = "ISO_10918_1";
		}
		else
		{
			retval = image->Encode(dptr, image_length, CXIMAGE_FORMAT_JP2);
			LossyImageCompressionMethod = "ISO_15444_1";
		}
		if (retval)
		{
			if (image_length)
			{
				ratio = (float)(BitsAllocated*Rows*Columns) / (float)image_length;
				char ttt[32];
				sprintf(ttt, "%f", ratio);
				LossyImageCompressionRatio = ttt;
			}
			else
				LossyImageCompressionRatio = "0";
		}
	}
	return retval;
}

#endif //USE_XIMAGE

#define DCM_CMD_SIZE 32768

CDICOMFile::CDICOMFile(void)
{	
	dicom_file=NULL;
	dicom_data_buffer=NULL;
	WL_BUF=NULL;
	dicom_buffer_size=0;
	dicom_file_size=0;
	DCM_cnt=0;
	status=0;
	DCM_CMD=(int8_t*)malloc(DCM_CMD_SIZE); // allocate buffer
	buf_command=NULL;
	IDX=0;
	//memset(m_items, 0, sizeof(m_items));
	m_items.clear();
}

CDICOMFile::~CDICOMFile(void)
{	
	if(dicom_file) 
		fclose(dicom_file);
	if(dicom_data_buffer) 
		free(dicom_data_buffer);
	if(WL_BUF) 
		free(WL_BUF);
	if(DCM_CMD) 
		free(DCM_CMD);
}

#define LUT_SIZE 4096

int8_t* CDICOMFile::GetImageWL(WL_t* params, int image_id) // see part 3 of standart
{	
	int16_t w=0,h=0;
	int i,len,lut_size=0;
	int bpp;
	short bstored;
	short padding;
	short representation;
	short	pix_id;
	LUT_descriptor_t lut;
	short* src;
	int8_t* src8;
	short* sdst;
	short sval;
	int mcnt=0;
	int8_t* dst;
	int8_t LUT[LUT_SIZE];
	uint16_t LUT16[LUT_SIZE];

	if(params==NULL) return NULL;
	if(WL_BUF) free(WL_BUF);
	GetDCMItem(0x00280010,&h);
	GetDCMItem(0x00280011,&w);
	GetDCMItem(0x00280101,&bstored);
	GetDCMItem(0x00280120,&padding); 
	GetDCMItem(0x00280103,&representation);
	bpp=GetDCMbpp();
	// pre create LUT
	for(i=0;i<LUT_SIZE;i++) 
	{	
		LUT[i]=(int8_t)i;
		LUT16[i]=i;
	}
	double intercept=params->intercept;
	double slope=params->slope;
	// handle slope, intercept
	double cd = (((double)params->center - intercept) / slope);
	double wd = ((double)params->width / slope);
	double k;
	double c0;
	double g = 0;
	if(params->lut==0) // create lut
	{	
		if(params->center!=0 && params->width!=0)
		{	
			//if(bpp==8) // create 8bit lut	
			{	
				k = 256.0/wd;
				c0 = (k * (cd - wd/2.0)) * -1.0;
				for(i=0;i<LUT_SIZE; i++)
				{	
					// get colorvalue
					g = k * i + c0;
					if(g > 255) 
					{	
						g = 255;
					}
					if(g < 0) 
					{	
						g = 0;
					}
					LUT[i]=(int8_t)g;
				}
			}
			/*			
			else
			{	k = 4096.0/wd;
				//k = 256.0/wd;
				c0 = (k * (cd - wd/2.0)) * -1.0;
				for(i=0;i<LUT_SIZE; i++)
				{	// get colorvalue
					g = k * i + c0;
					if(g > 4095) 
					{	g = 4095;
					}
					if(g < 0) 
					{	g = 0;
					}
					LUT16[i]=(uint16_t)g;
				}

			}
			*/
		}
	}
	else // get LUT from file
	{	
		if(GetLUT((int8_t*)&LUT16[0],&lut)) lut_size=lut.size;
		else 
		{	
			lut_size=0;
			params->lut=0;
		}
	}
	// create new buffer
	len=w*h;
	//WL_BUF=(int8_t*)malloc(len*(params->lut+1)+32);
	WL_BUF=(int8_t*)malloc(len+32);
	if(!WL_BUF) return NULL;
	src=(short*)GetImagePtr( image_id);
	src8=(int8_t*)src;
	if(src==NULL) return NULL;
	dst=WL_BUF;
	sdst=(short*)WL_BUF;
	/* calculations
	if (x <= c - 0.5 - (w-1)/2), then y = ymin  // v1
	else if (x > c - 0.5 + (w-1)/2), then y = ymax, // v2
	else y = ((x - (c - 0.5)) / (w-1) + 0.5) * (ymax - ymin)+ ymin
	*/
	double v1=(double)params->center-0.5-((double)params->width-1)/2;
	double v2=(double)params->center-0.5+((double)params->width-1)/2;
	for(i=0;i<len;i++)
	{	
		if(params->lut)
		{	
			pix_id=*src;
			pix_id-=lut.first_val;
			if(pix_id<0) {sval=LUT16[0];}
			else if (pix_id<(int)lut.size) { sval=LUT16[pix_id];}
			else sval=LUT16[lut.size-1];
			*dst = (int8_t)(sval >> 4);
			//*sdst++;
			*dst++;
			*src++;
		}
		else 
		{	
			if(bpp==16)
			{	
				sval=*src;
				if(*src==(short)params->padding) *dst=0;
				else
				{	
					if(representation==1) 
					{	
						//sval-=params->padding;
						if((double)sval<v1) *dst=0;
						else if((double)sval>v2) *dst=255;
						else *dst=(int8_t)((((double)sval-((double)params->center-0.5))/((double)params->width-1)+0.5)*255);
					}
					else *dst=LUT16[sval&0xFFF]>>4;
				}
				*src++;
			}
			else
			{	
				//if(*src8==(byte)params->padding) *dst=0;
   				*dst=LUT[*src8&0xFF];
				*src8++;
			}
			*dst++;
		}
	}
	return WL_BUF; 
}

#define EXTRA_DATA_TAIL 32768

// returns false on error
// mode 
// 0 - autodetect reduced DCM
// 1 - offline storage header exists
bool CDICOMFile::OpenDICOMFile(char *path,int mode)
{	
	bool retval=false;
	int8_t* ptr;
	uint32_t z;
	char buf[16];
	uint64_t  read_size=0;

	file_name.clear();
	is_explicit=false;
	if(dicom_data_buffer) 
		free(dicom_data_buffer);
	dicom_file = fopen( path, "rb");
	if(dicom_file==NULL) 
		return false;
	fseek( dicom_file, 0, SEEK_SET);
	fseek( dicom_file, 0, SEEK_END);
	dicom_file_size = ftell( dicom_file);
	if(dicom_file_size<=0)
	{	
		fclose(dicom_file);
		return false;
	}
	if(!mode) // autodetect RDCM
	{	
		fseek( dicom_file,0,SEEK_SET);
		fread( buf,8,1,dicom_file);
		if( buf[0]) mode=1;
	}
	//if(!mode) BUF_SIZE-=132; //
	dicom_buffer_size = dicom_file_size;//*2+EXTRA_DATA_TAIL;// add extra data buffer
	dicom_data_buffer = (int8_t*)malloc(dicom_buffer_size);
	if(!dicom_data_buffer)
	{	
		fclose(dicom_file);
		return false;
	}
	memset(buf, 0, 16);
	memset(dicom_data_buffer, 0, dicom_buffer_size);
	if(!mode)
	{	
		read_size=dicom_file_size-128L;
		if(read_size>0)
		{
			if(!fseek(dicom_file, 128, SEEK_SET))
			{	
				fread(buf,4,1,dicom_file);
				if(!strncmp(buf, DICOM_FILE_TAG, 4)) retval=true;
				read_size-=4;
			}
		}
		if(!retval)
		{	
			if(dicom_file) fclose(dicom_file);
			return retval;
		}
	}
	else 
	{
		fseek(dicom_file,0,SEEK_SET); // skip PDV header
		read_size=dicom_file_size;
	}
	ptr=dicom_data_buffer;
	z=fread(ptr, read_size, 1, dicom_file);
	if(z)
	{
		if(*(ptr+5)) is_explicit=true;
		if(is_explicit) mode=0; // !!!!!!!!!!
		dicom_buffer_size=read_size;
		retval=LoadFile(mode);
		file_name=path;
	}
	else 
		retval=false;
	if(dicom_file) 
		fclose(dicom_file);
	return retval;
}

bool CDICOMFile::Init()
{	
	// TBD clear memeory
	m_items.clear();
	v_Image_pointers.clear();
	v_Image_lengths.clear();
	DIR_ptr=NULL;
	dicom_buffer_size=0;
	dicom_file_size=0;
	// init internal dict
	return true;
}

bool CDICOMFile::SaveDICOMFile(char* filename, int mode/* =DCM_SAVE_FULL */)
{	
	bool retval=false;
	//CString outfilename;
	FILE* outfile=NULL;
	char z[128];

	outfile=fopen(filename, "wb");
	if(outfile==NULL) return retval;
	// save preamble
	if(mode==DCM_SAVE_FULL)
	{
		memset(z, 0, 128);
		fwrite(&z, 128, 1, outfile);
		fwrite( DICOM_FILE_TAG, strlen(DICOM_FILE_TAG), 1, outfile);
	}
	// save tags
	retval=SaveTags(0x00020000, outfile);
	if(retval) retval=SaveTags(0x00080000, outfile);
	if(retval) retval=SaveTags(0x00100000, outfile);
	if(retval) retval=SaveTags(0x00180000, outfile);
	if(retval) retval=SaveTags(0x00200000, outfile);
	if(retval) retval=SaveTags(0x00280000, outfile);
	if(retval) retval=SaveTags(0x00320000, outfile);
	// WSI
	if (retval) retval = SaveTags(0x00400000, outfile);
	if (retval) retval = SaveTags(0x00480000, outfile);
	if (retval) retval = SaveTags(0x52000000, outfile);
	// images
	if(retval) retval = SaveTags(0x7FE00010, outfile);
	if (retval) retval = SaveTags(0xFFFEE000, outfile);
	//if (retval) retval = SaveTags(0xFFFEE0DD, outfile);
	fflush(outfile);
	fclose(outfile);
	return retval;
}


// load tags into data buffer
bool CDICOMFile::LoadFile(int mode)
{	
	int8_t* ptr;
	bool skip=false;
	bool dir_on=false;
	bool add_image=false;
	DATA_SET* ds = NULL;
	DATA_SET_C* ds_r = NULL;
	DATA_SET_B* ds_b = NULL;
	DATA_SET_C* ds_c = NULL;
	uint16_t t;
	int64_t z;
	int size,len;
	int pre_mode=mode;
	
	ptr=dicom_data_buffer;
	for(DCM_cnt=0,z=0;(DCM_cnt<m_items.size()) && (z<dicom_buffer_size);DCM_cnt++)
	{	
		mode=pre_mode;
		add_image=false;
		if(!pre_mode)
		{	
			ds=(DATA_SET*)ptr;
			if(!dir_on)
			{	
				if(ds->vr<VR_DA) mode=1;
				else if(ds->vr>VR_OW) mode=1;
			}
		}
		if(!mode)	//dicom dir
		{	
			ds=(DATA_SET*)ptr;
			t=ds->tag.val[0];
			ds->tag.val[0]=ds->tag.val[1];
			ds->tag.val[1]=t;
			m_items[DCM_cnt].tag=ds->tag;
		}
		else
		{	
			ds_r=(DATA_SET_C*)ptr;
			t=ds_r->tag.val[0];
			ds_r->tag.val[0]=ds_r->tag.val[1];
			ds_r->tag.val[1]=t;
			m_items[DCM_cnt].tag=ds_r->tag;
		}
		skip=false;
		m_items[DCM_cnt].item_ptr=ptr;
		if(!mode)
		{	
			if ( ds)
			{
				if(ds->tag.tag==0x00041220) 
				{	
					DIR_ptr=m_items[DCM_cnt].data_ptr;
					dir_on=true;
				}
				switch(ds->tag.tag)
				{	
					case 0xFFFAFFFA :
					case 0xFFFEE000 : 
					case 0xFFFEE00D : 
					case 0xFFFEE0DD :	m_items[DCM_cnt].vr=VR_SQ;
										ds_c=(DATA_SET_C*)ptr;
										len=ds_c->val_size;
										if(len==0xFFFFFFFF) len=0;
										if(!dir_on)
										{	
											size=sizeof(DATA_SET_C)+len;
											skip=true;
										}
										else 
											size=sizeof(DATA_SET_C);
										break; 
					default : break;
				}
			}
		}
		else
		{
		}
		if(skip==false)
		{	
			if(!mode)
			{	
				switch(ds->vr)
				{	
					case VR_OB :
					case VR_OW :
					case VR_SQ :
					case VR_UN :	ds_b=(DATA_SET_B*)ptr;
		                 len=ds_b->val_size;
		  			     if(len==0xFFFFFFFF) len=0;
						 if(dir_on) len=0;
		                 size=sizeof(DATA_SET_B)+len;
					     break;
					default :	len=ds->val_size;
								size=sizeof(DATA_SET)+len;
								break;
				}
				m_items[DCM_cnt].vr=ds->vr;
			}
			else
			{	
				ds_c=(DATA_SET_C*)ptr;
				len=ds_c->val_size;
				if(len==0xFFFFFFFF) len=0;
				size=sizeof(DATA_SET_C)+len;
				m_items[DCM_cnt].vr=GetItemVR(ds_c->tag.tag);
			}
		}
		m_items[DCM_cnt].size=len;
		m_items[DCM_cnt].data_ptr=(ptr+size-len);
		if(!mode)
		{	
			if(ds->tag.tag==0x7FE00010) 
			{	
				add_image=true;
			}
			else if(ds->tag.tag==0xFFFEE000) 
			{	
				add_image=true;
			}
		}
		else
		{	
			if (ds_r)
			{
				if (ds_r->tag.tag == 0x7FE00010)
				{
					add_image = true;
				}
				else if (ds_r->tag.tag == 0xFFFEE000)
				{
					add_image = true;
				}
				if (ds_r->tag.tag == 0x00041220)
				{
					DIR_ptr = m_items[DCM_cnt].data_ptr;
				}
			}
		}
		if(add_image)
		{
			v_Image_pointers.push_back(m_items[DCM_cnt].data_ptr);
			v_Image_lengths.push_back(m_items[DCM_cnt].size);
		}
		ptr+=size;
		z+=size;
	}
	return true;
}

char* CDICOMFile::GetDCMItemTag_T(int idx)
{ 
	memset(m_string,0,sizeof(m_string));
  	sprintf(m_string,"%8.8X",m_items[idx].tag.tag);
	return m_string;
}

char* CDICOMFile::GetDCMItemVal_T(int xidx)
{ 
	uint32_t v=0L;
	float fv=0;
	double fvd=0;
	int idx;

	if(xidx<m_items.size()) 
		idx=xidx;
	else
	{ 
		if(GetDCMItem(xidx,&idx)) 
			idx=IDX;
		else 
			return NULL;
	}
	memset(m_string,0,sizeof(m_string));
	switch(m_items[idx].vr)
	{ 
		case VR_AE :
		case VR_AS :
		case VR_CS :
		case VR_DA :
		case VR_DS :
		case VR_DT :
		case VR_IS :
		case VR_LO :
		case VR_LT :
		case VR_PN :
		case VR_SH :
		case VR_ST :
		case VR_TM :
		case VR_UI : memcpy(m_string, m_items[idx].data_ptr, m_items[idx].size); break;
		case VR_SS: v = *(int16_t*)m_items[idx].data_ptr; _itoa(v, m_string, 10); break;
		case VR_US: v = *(uint16_t*)m_items[idx].data_ptr; _itoa(v, m_string, 10); break;
		case VR_SL: v = *(int32_t*)m_items[idx].data_ptr; _ltoa(v, m_string, 10); break;
		case VR_UL: v = *(uint32_t*)m_items[idx].data_ptr; _ultoa(v, m_string, 10); break;
		case VR_FD: fvd = *(double*)m_items[idx].data_ptr; sprintf(m_string, "%f", fvd); break;
		case VR_FL: fv = *(float*)m_items[idx].data_ptr; sprintf(m_string, "%f", fv); break;
		case VR_OB :
		case VR_SQ :
		case VR_OW: v = m_items[idx].size; sprintf(m_string, "Len[%d]", v); break;
		default : break;
	}
	return m_string;
}

char* CDICOMFile::GetDCMItemVR_T(int idx)
{ 
	uint16_t* ptr=(uint16_t*)&m_string[0];
  	
	memset(m_string,0,sizeof(m_string));
	if (idx<m_items.size())
		*ptr = m_items[idx].vr;
	return m_string;
}

char* CDICOMFile::GetDCMItemDescription(int idx)
{ 
	int i;
	char* retval=NULL;
	
	if(idx<m_items.size())
	{
		for(i=0;DICOM_DICT[i].vr!=VR_END;i++)
		{ 
			if (m_items[idx].tag.tag == DICOM_DICT[i].tag) 
				return (char*)DICOM_DICT[i].name;
		}
	}
	return "Unknown tag";
}

int CDICOMFile::GetIntVal(char* ptr)
{ 
	char ttt[64];
	int i;
	char* z=ptr;

	memset(ttt,0,64);
	for(i=0;(i<64) && *z;i++,*z++)
	{ 
		if(*z=='-') 
			ttt[i]=*z;
		else if(isdigit((unsigned char)*z))
			ttt[i]=*z;
		else 
			break;
	}
	return atoi(ttt);
}

#define COLOR_MONO "MONOCHROME"

bool CDICOMFile::GetWL(WL_t* wlt)
{	
	char* param;
	unsigned short usval;
	bool retval=false;

	wlt->lut=0;
	if(GetDCMItem(0x00280004,(int8_t**)&param))
	{	
		if(!_strnicmp(param,COLOR_MONO,strlen(COLOR_MONO))) retval=true;
		else retval=false;
	}
	retval=HasWL();
	if(retval)
	{	
		// get values
		if(GetDCMItem(0x00280120,&usval)) wlt->padding=usval; 
		else wlt->padding=65535;
		if(GetDCMItem(0x00281050,(int8_t**)&param)) wlt->center=GetIntVal(param); 
		else 
		{	
			wlt->center=0;
			wlt->lut=1;
		}
		if (GetDCMItem(0x00281051, (int8_t**)&param)) wlt->width = GetIntVal(param);
		else wlt->width=1;
		if (GetDCMItem(0x00281053, (int8_t**)&param))
		{	
			wlt->slope=GetIntVal(param); 
			if(wlt->slope==0) wlt->slope=1;
		}
		else wlt->slope=1;
		if (GetDCMItem(0x00281052, (int8_t**)&param))
		{	
			wlt->intercept=GetIntVal(param); 
			//if(wlt->intercept==0) wlt->intercept=1;
		}
		else wlt->intercept=0;
	}
	return retval;
}

bool CDICOMFile::HasWL()
{	
	char* param;
	bool retval=false;
	
	if (GetDCMItem(0x00280004, (int8_t**)&param))
	{	
		if(!_strnicmp(param,COLOR_MONO,strlen(COLOR_MONO))) retval=true;
		else retval=false;
	}
	return retval;
}

VR CDICOMFile::GetItemVR(uint32_t tag)
{	
	int i;
	
	for (i = 0; DICOM_DICT[i].vr != VR_END; i++)
	{	
		if (tag == DICOM_DICT[i].tag)
			return (VR)DICOM_DICT[i].vr;
	}
	return VR_END;
}

int CDICOMFile::GetItemLength(int idx)
{	
	if(idx>=DCM_cnt) 
		return 0;
	return m_items[idx].size;
}

bool CDICOMFile::CloseDICOMFile()
{	
	Init();
	return true;
}

int CDICOMFile::GetDCM_calib_x()
{ 
	int8_t* ptr=NULL;
	int i;
	char stg[80];
	int retval=0;
	double f;
	
	GetDCMItem(0x00280030,&ptr);
	if(ptr)
	{ 
		memset(stg,0,80); 
		memcpy(stg,ptr,64);
		for(i=0;i<80;i++)
		{ 
			if(stg[i]==(int8_t)'\\') break;
		}
		memset(stg+i,0,80-i);
		f=atof(stg);
		i=(int)(f*1000);
		return i;
	}
	else return 0;
}

int CDICOMFile::GetDCM_calib_y()
{ 
	int8_t* ptr=NULL;
	int i;
	char stg[80];
	int retval=0;
	double f;

	GetDCMItem(0x00280030,&ptr);
	if(ptr)
	{ 
		memset(stg,0,80); 
		memcpy(stg,ptr,64);
		for(i=0;i<80;i++)
		{ 
			if(stg[i]==(int8_t)'\\') break;
		}
		i++;
		f=atof(stg+i);
		i=(int)(f*1000);
		return i;
	}
	else return 0;
}

uint32_t CDICOMFile::GetImagesCount()
{
	uint16_t z=0;
	int retval=0, cnt;
	int8_t* val=NULL;

	GetDCMItem(0x00280008, &val);
	if(val) 
		retval=atoi( (char*)val);
	cnt=(uint32_t)v_Image_lengths.size();
	if( retval==0) 
		retval=cnt;
	else
	{
		if(retval!=cnt) 
			retval=cnt;
	}
	return retval;
}

// returns bits per pixel
int CDICOMFile::GetDCMbpp()
{	
	uint16_t data=0;
	uint16_t z=0;
	
	GetDCMItem(0x00280002,&z);
	if(!z) 
		z=1;
	GetDCMItem(0x00280100,&data);
	if(!data) 
		data=8;
	return (int)(data*z);
}

int CDICOMFile::GetDCMCompression(int* is_lossless)
{ 
	int8_t* ptr=NULL;
	char stg[80];
	char stg1[80];
	int retval=0, lossless=0;
	int8_t* p;
  
	
	memset(stg,0,80);
	memset(stg1,0,80);
	GetDCMItem(0x00282110,&ptr);
	if(ptr)
	{ 
		//memset(stg1,0,80);
		memcpy(stg1,ptr,64);
	}
	ptr=NULL;
	GetDCMItem(0x00020010,&ptr);
	if(ptr)
	{ 
		//memset(stg,0,80);
		memcpy(stg,ptr,64);
	}
	p=(int8_t*)stg;
	while(*p++!=0x00) 
	{ 
		if(*p<0x20) *p=0;
	}
	//if(atoi(stg1)>0) {retval=1; compression_name="JPEG Lossy";}
	if(!strcmp("1.2.840.10008.1.2",stg)) {retval=0; compression_name="No compression";}
	else if(!strcmp("1.2.840.10008.1.2.1",stg))  { retval=0; compression_name="No compression";}
	else if(!strcmp("1.2.840.10008.1.2.1.99",stg))  { retval=0; compression_name="No compression";}
	else if(!strcmp("1.2.840.10008.1.2.2",stg))  { retval=0; compression_name="No compression";}
	else if(!strcmp("1.2.840.10008.1.2.4.50",stg))  { retval=1; compression_name="JPEG Baseline (Process 1)";}
	else if(!strcmp("1.2.840.10008.1.2.4.51",stg))  { retval=2; compression_name="JPEG Extended (Process 2 & 4)";}
	else if(!strcmp("1.2.840.10008.1.2.4.52",stg))  { retval=3; compression_name="JPEG Extended (Process 3 & 5)";}
	else if(!strcmp("1.2.840.10008.1.2.4.53",stg))  { retval=4; compression_name="JPEG Spectral Selection Non-Hierarchical (Process 6 & 8)";}
	else if(!strcmp("1.2.840.10008.1.2.4.54",stg))  { retval=5; compression_name="JPEG Spectral Selection Non-Hierarchical (Process 7 & 9)";}
	else if(!strcmp("1.2.840.10008.1.2.4.55",stg))  { retval=6; compression_name="JPEG Full Progression Non-Hierarchical (Process 10 & 12)";}
	else if(!strcmp("1.2.840.10008.1.2.4.56",stg))  { retval=7; compression_name="JPEG Full Progression Non-Hierarchical (Process 11 & 13)";}
	else if(!strcmp("1.2.840.10008.1.2.4.57",stg))  { retval=8; compression_name="JPEG Lossless Non-Hierarchical(Process 14)"; lossless=1;}
	else if(!strcmp("1.2.840.10008.1.2.4.58",stg))  { retval=9; compression_name="JPEG Lossless Non-Hierarchical (Process 15)"; lossless=1;}
	else if(!strcmp("1.2.840.10008.1.2.4.59",stg))  { retval=10; compression_name="JPEG Extended Hierarchical (Process 16 & 18)";}
	else if(!strcmp("1.2.840.10008.1.2.4.60",stg))  { retval=11; compression_name="JPEG Extended Hierarchical (Process 17 & 19)";}
	else if(!strcmp("1.2.840.10008.1.2.4.61",stg))  { retval=12; compression_name="JPEG Spectral Selection Hierarchical (Process 20 & 22)";}
	else if(!strcmp("1.2.840.10008.1.2.4.62",stg))  { retval=13; compression_name="JPEG Spectral Selection Hierarchical (Process 21 & 23)";}
	else if(!strcmp("1.2.840.10008.1.2.4.63",stg))  { retval=14; compression_name="JPEG Full Progression Hierarchical (Process 24 & 26)";}
	else if(!strcmp("1.2.840.10008.1.2.4.64",stg))  { retval=15; compression_name="JPEG Full Progression Hierarchical (Process 25 & 27)";}
	else if(!strcmp("1.2.840.10008.1.2.4.65",stg))  { retval=16; compression_name="JPEG Lossless Hierarchical (Process 28)"; lossless=1;}
	else if(!strcmp("1.2.840.10008.1.2.4.66",stg))  { retval=17; compression_name="JPEG Lossless Hierarchical (Process 29) "; lossless=1;}
	else if(!strcmp("1.2.840.10008.1.2.4.70",stg))  { retval=18; compression_name="JPEG Lossless Non-Hierarchical First-Order Prediction (Process 14)"; lossless=1;}
	else if(!strcmp("1.2.840.10008.1.2.4.80",stg))  { retval=19; compression_name="JPEG-LS Lossless"; lossless=1;}
	else if(!strcmp("1.2.840.10008.1.2.4.81",stg))  { retval=20; compression_name="JPEG-LS Lossy (Near-Lossless)";}
	else if(!strcmp("1.2.840.10008.1.2.4.90",stg))  { retval=21; compression_name="JPEG 2000 Image Compression (Lossless Only)";}
	else if(!strcmp("1.2.840.10008.1.2.4.91",stg))  { retval=22; compression_name="JPEG 2000 Image Compression";}
	else if(!strcmp("1.2.840.10008.1.2.5",stg))  { retval=30; compression_name="RLE Lossless";}
	else { retval=-1; compression_name="Unknown";}
	if(is_lossless)
	{
		*is_lossless=lossless;
	}
	return retval;
}

char* CDICOMFile::GetDCMCompressionName()
{	
	return ( char*)compression_name.c_str();
}

void CDICOMFile::InitCommand()
{	
	dcm_cmd_length=0;
}

//extern LPCTSTR DB_header[];

bool CDICOMFile::IsDICOMDir()
{	
	int z=0;
	bool retval;
	retval=GetDCMItem(0x00041220,&z); 
	return retval;
}

bool CDICOMFile::GetLUT(int8_t* lut,LUT_descriptor_t* lut_descr)
{	
	int8_t* src; 
	int8_t* bptr;
	DATA_SET* ptr;
	LUT_descriptor_t* ltd;
	int z=0,len=0;
	
	if(!GetDCMItem(0x00283010,&src,z)) 
		return false;
	len=GetItemLength(IDX);
	bptr=NULL;
	z=8;
	do
	{	
		ptr=(DATA_SET*)(src+z);
		if(is_explicit) 
			z+=(ptr->val_size+8);
		else 
			z+=(ptr->vr+8);
		if(ptr->tag.tag==0x30020028)
		{	
			bptr = (int8_t*)ptr;
			bptr+=8;
			ltd=(LUT_descriptor_t*)bptr;
			memcpy(lut_descr,ltd,sizeof(LUT_descriptor_t));
			bptr=NULL;
		}
		else if(ptr->tag.tag==0x30060028)
		{	
			if(is_explicit) 
				len=ptr->val_size;
			else 
				len=ptr->vr;
			bptr = (int8_t*)ptr;
			bptr+=8;
			break;
		}
	}  
	while(z<len);
	if(bptr) 
	{	
		memcpy(lut, bptr, len);
	}
	return true;
}

// 
bool CDICOMFile::Convert2Send(char* path)
{	
	int i;
	FILE* file;
	unsigned int tag;

	if(path==NULL) 
		return false;
	file=fopen(path,"wb");
	if(!file) 
		return false;
	fseek(file,0,SEEK_SET);
	for(i=0;i<DCM_cnt;i++)
	{	
		if (m_items[i].tag.val[1] == 0x0002) 
			continue; // skip off-line info
		tag = sw(m_items[i].tag.tag);
		fwrite(&tag,sizeof(int),1,file);
		fwrite(&m_items[i].size, sizeof(int), 1, file);
		fwrite(m_items[i].data_ptr, m_items[i].size, 1, file);
	}
	fflush(file);
	fclose(file);
	return true;
}

FILE* CDICOMFile::Convert2SendEx(char* path)
{	
	int i;
	FILE* file=NULL;
	unsigned int tag;

	if(path==NULL) return file;
	file=fopen(path, "w+b");
	if(!file) return false;
	fseek(file, 0, SEEK_SET);
	for(i=0; i<DCM_cnt; i++)
	{	
		if (m_items[i].tag.val[1] == 0x0002)
			continue; // skip off-line info
		tag = sw(m_items[i].tag.tag);
		fwrite(&tag, sizeof(int), 1, file);
		fwrite(&m_items[i].size, sizeof(int), 1, file);
		fwrite(m_items[i].data_ptr, m_items[i].size, 1, file);
	}
	fflush(file);
	fseek(file, 0, SEEK_SET);
	return file;
}
/*
int	ReadJpegData( char *buffer, int numBytes)
{
	int rest, retval=0;
	//return fread(buffer, 1, numBytes, inFile);
	
	rest=IMA_len-IMA_off;
	if(rest<=0) return retval;
	if(numBytes>rest) retval=rest;
	else retval=numBytes;
	memcpy( buffer, IMA_ptr+IMA_off, retval);
	IMA_off+=retval;
	return retval;
}
*/

// item handling

bool CDICOMFile::GetDCMItem(int tag,uint16_t *data,int next)
{ 
	int i;
	bool retval=false;

	i=FindItemIdx(tag, next);
	if(i>=0)
	{
		retval=true;
		*data = *(uint16_t*)m_items[i].data_ptr;
	}
	return retval;
}

bool CDICOMFile::GetDCMItem(int tag,int16_t *data,int next)
{ 
	int i;
	bool retval=false;

	i=FindItemIdx(tag, next);
	if(i>=0)
	{ 
		retval=true;
		*data = *(int16_t*)m_items[i].data_ptr;
	}
	return retval;
}

bool CDICOMFile::GetDCMItem(int tag,uint32_t *data,int next)
{	
	int i;
	bool retval=false;

	i=FindItemIdx(tag, next);
	if(i>=0)
	{ 
		retval=true;
		*data = *(uint32_t*)m_items[i].data_ptr;
	}
	return retval;
}

bool CDICOMFile::GetDCMItem(int tag,int32_t *data,int next)
{	
	int i;
	bool retval=false;

	i=FindItemIdx(tag, next);
	if(i>=0)
	{ 
		retval=true;
		*data = *(int32_t*)m_items[i].data_ptr;
	}
	return retval;
}

bool CDICOMFile::GetDCMItem(int tag,float *data,int next)
{	
	int i;
	bool retval=false;

	i=FindItemIdx(tag, next);
	if(i>=0)
	{ 
		retval=true;
		*data = *(float*)m_items[i].data_ptr;
	}
	return retval;
}

bool CDICOMFile::GetDCMItem(int tag,double *data,int next)
{	
	int i;
	bool retval=false;

	i=FindItemIdx(tag, next);
	if(i>=0)
	{ 
		retval=true;
		*data = *(double*)m_items[i].data_ptr;
	}
	return retval;
}

bool CDICOMFile::GetDCMItem(int tag,int8_t **data,int next)
{	
	int i;
	bool retval=false;

	i=FindItemIdx(tag, next);
	if(i>=0)
	{ 
		retval=true;
		*data = m_items[i].data_ptr;
	}
	return retval;
}

char* CDICOMFile::GetDCMItemEx(int tag,int next)
{	
	int i;
	char* retval=NULL;

	i=FindItemIdx(tag, next);
	if(i>=0)
	{ 
		retval = (char*)m_items[i].data_ptr;
	}
	return retval;
}

bool CDICOMFile::SetDCMItem(int tag, uint16_t *data)
{
	int i;
	bool retval=false;

	i=FindItemIdx(tag);
	if (i >= 0 && m_items[i].data_ptr)
	{
		retval=true;
		*(uint16_t*)m_items[i].data_ptr = *data;
		m_items[i].flags |= DCM_DATA_FLAG_MODIFIED;
	}
	return retval;
}

bool CDICOMFile::SetDCMItem(int tag, int16_t *data)
{		
	int i;
	bool retval=false;

	i=FindItemIdx(tag);
	if (i >= 0 && m_items[i].data_ptr)
	{
		retval=true;
		*(int16_t*)m_items[i].data_ptr = *data;
		m_items[i].flags |= DCM_DATA_FLAG_MODIFIED;
	}
	return retval;
}

bool CDICOMFile::SetDCMItem(int tag, uint32_t *data)
{
	int i;
	bool retval=false;

	i=FindItemIdx(tag);
	if (i >= 0 && m_items[i].data_ptr)
	{
		retval=true;
		*(uint32_t*)m_items[i].data_ptr = *data;
		m_items[i].flags |= DCM_DATA_FLAG_MODIFIED;
	}
	return retval;
}

bool CDICOMFile::SetDCMItem(int tag, int32_t *data)
{
	int i;
	bool retval=false;

	i=FindItemIdx(tag);
	if (i >= 0 && m_items[i].data_ptr)
	{
		retval=true;
		*(int32_t*)m_items[i].data_ptr = *data;
		m_items[i].flags |= DCM_DATA_FLAG_MODIFIED;
	}
	return retval;
}

bool CDICOMFile::SetDCMItem(int tag, float *data)
{
	int i;
	bool retval=false;

	i=FindItemIdx(tag);
	if (i >= 0 && m_items[i].data_ptr)
	{
		retval=true;
		*(float*)m_items[i].data_ptr = *data;
		m_items[i].flags |= DCM_DATA_FLAG_MODIFIED;
	}
	return retval;
}

bool CDICOMFile::SetDCMItem(int tag, double *data)
{
	int i;
	bool retval=false;

	i=FindItemIdx(tag);
	if (i >= 0 && m_items[i].data_ptr)
	{
		retval=true;
		*(double*)m_items[i].data_ptr = *data;
		m_items[i].flags |= DCM_DATA_FLAG_MODIFIED;
	}
	return retval;
}

bool CDICOMFile::SetDCMItem(int tag, int8_t **data)
{
	int i;
	bool retval=false;

	i=FindItemIdx(tag);
	if(i>=0)
	{ 
		retval=true;
		m_items[i].data_ptr = *data;
		m_items[i].flags |= DCM_DATA_FLAG_MODIFIED;
	}
	return retval;
}

bool CDICOMFile::SetDCMItem(int tag, char *data)
{
	int len;
	bool retval=false;

	if(data==NULL) 
		return retval;
	len=strlen(data);
	return SetDCMItem(tag, data, len);
}

bool CDICOMFile::SetDCMItem(int tag, void *data, int data_len)
{
	int i, len;
	bool retval=false;

	if(data==NULL) return retval;
	len=data_len;
	i=FindItemIdx(tag);
	if(i>=0)
	{ 
		retval=true;
		if(len==0)
		{
			if (m_items[i].data_ptr) 
				*m_items[i].data_ptr = 0;
		}
		else
		{	
			if (len>m_items[i].size)
			{	
				if (m_items[i].flags&DCM_DATA_FLAG_ALLOCATED)
				{
					if (m_items[i].data_ptr) 
						free(m_items[i].data_ptr);
					m_items[i].data_ptr = (int8_t*)malloc(len + 1);
					if (m_items[i].data_ptr) 
						m_items[i].flags |= DCM_DATA_FLAG_ALLOCATED;
					else retval=false;
				}
			}
			if (m_items[i].data_ptr) 
				memcpy(m_items[i].data_ptr, data, len);
		}
		if(retval)
		{
			m_items[i].flags |= DCM_DATA_FLAG_MODIFIED;
			m_items[i].size = len;
		}
	}
	return retval;
}

int8_t* CDICOMFile::FillDCMItem(int& idx, int tag, int len, tag_list& _items)
{
	DCM_DATA item;
	int xlen = len;

	if (idx < 0) // create new item
	{
		item.data_ptr = NULL;
		item.item_ptr = NULL;
		item.flags = 0;
		_items.push_back(item);
		idx = _items.size() - 1;
	}

	if (_items[idx].data_ptr && (_items[idx].flags&DCM_DATA_FLAG_ALLOCATED))
	{
		free(_items[idx].data_ptr);
	}
	if (len < 0)
		xlen = 0;
	_items[idx].data_ptr = (int8_t*)malloc(xlen);
	if (_items[idx].data_ptr == NULL)
		return NULL;
	_items[idx].size = len;
	_items[idx].flags |= DCM_DATA_FLAG_MODIFIED;
	_items[idx].flags |= DCM_DATA_FLAG_ALLOCATED;
	_items[idx].item_ptr = (int8_t*)&_items[idx];
	_items[idx].tag.tag = tag;
	_items[idx].vr = GetItemVR(tag);
	_items[idx].type = 0;
	return _items[idx].data_ptr;
}

bool CDICOMFile::AddDCMItem(int tag, void* data, int datalen, tag_list& _items)
{
	int i = -1, len;
	bool retval=false;

	//if(data==NULL && datalen!= -1) 
	//	return retval;
	len=datalen;
	if(FillDCMItem(i, tag, len, _items))
	{
		retval=true;
		_items[i].data_ptr = (int8_t*)data;
	}
	return retval;
}


bool CDICOMFile::AddDCMItem(int tag, uint16_t *data, tag_list& _items)
{
	int i = -1, len;
	bool retval=false;

	if(data==NULL) 
		return retval;
	len=sizeof(uint16_t);
	if(FillDCMItem(i, tag, len, _items))
	{
		retval=true;
		*(uint16_t*)_items[i].data_ptr = *data;
	}
	return retval;
}

bool CDICOMFile::AddDCMItem(int tag, int16_t *data, tag_list& _items)
{
	int i = -1, len;
	bool retval=false;

	if(data==NULL) 
		return retval;
	len=sizeof(int16_t);
	if(FillDCMItem(i, tag, len, _items))
	{
		retval=true;
		*(int16_t*)_items[i].data_ptr = *data;
	}
	return retval;
}

bool CDICOMFile::AddDCMItem(int tag, uint32_t *data, tag_list& _items)
{
	int i = -1, len;
	bool retval=false;

	if(data==NULL) 
		return retval;
	len=sizeof(uint32_t);
	if(FillDCMItem(i, tag, len, _items))
	{
		retval=true;
		*(uint32_t*)_items[i].data_ptr = *data;
	}
	return retval;
}

bool CDICOMFile::AddDCMItem(int tag, int32_t *data, tag_list& _items)
{
	int i = -1, len;
	bool retval=false;

	if(data==NULL) 
		return retval;
	len=sizeof(int32_t);
	if(FillDCMItem(i, tag, len, _items))
	{
		retval=true;
		*(int32_t*)_items[i].data_ptr=*data;
	}
	return retval;
}

bool CDICOMFile::AddDCMItem(int tag, float *data, tag_list& _items)
{
	int i = -1, len;
	bool retval=false;

	if(data==NULL) 
		return retval;
	len=sizeof(float);
	if(FillDCMItem(i, tag, len, _items))
	{
		retval=true;
		*(float*)_items[i].data_ptr=*data;
	}
	return retval;
}

bool CDICOMFile::AddDCMItem(int tag, double *data, tag_list& _items)
{
	int i = -1, len;
	bool retval=false;

	if(data==NULL) 
		return retval;
	len=sizeof(double);
	if(FillDCMItem(i, tag, len, _items))
	{
		retval=true;
		*(double*)_items[i].data_ptr=*data;
	}
	return retval;
}

bool CDICOMFile::AddDCMItem(int tag, int8_t **data, tag_list& _items)
{
	int i = -1, len;
	bool retval=false;

	if(data==NULL) 
		return retval;
	len=sizeof(int8_t*);
	if(FillDCMItem(i, tag, len, _items))
	{
		retval=true;
		_items[i].data_ptr=*data;
	}
	return retval;
}

bool CDICOMFile::AddDCMItem(int tag, const char *data, tag_list& _items)
{
	int i = -1, len;
	bool retval=false;

	if(data==NULL) 
		return retval;
	len=strlen(data);
	if(len&0x01) 
		len++; // even
	if(FillDCMItem(i, tag, len, _items))
	{
		retval=true;
		memcpy(_items[i].data_ptr, data, len);
	}
	return retval;
}

// dicom file
bool CDICOMFile::Convert2Send(char* buffer, int max_len)
{
	int i;

	if(buffer==NULL) return false;
	for(i=0; i<DCM_cnt; i++)
	{	
		if(m_items[i].tag.val[1]==0x0002) 
			continue; // skip off-line info
		//tag=sw(m_items[i].tag.tag);
		//fwrite(&tag,sizeof(int),1,file);
		//fwrite(&m_items[i].size,sizeof(int),1,file);
		//fwrite(m_items[i].data_ptr,m_items[i].size,1,file);
	}
	return true;
}

int CDICOMFile::FindItemIdx(int tag, int startfrom)
{
	int i;

	for(i=startfrom; i<m_items.size(); i++) 
	{	
		if(m_items[i].tag.tag==tag) 
		{
			IDX=i;  // set last found 
			return i;
		}
	}
	return -1;
}

int CDICOMFile::FindItemIdx(int tag, tag_list& _items)
{
	int i;

	for (i = 0; i < _items.size(); i++)
	{
		if (_items[i].tag.tag == tag)
		{
			//IDX = i;  // set last found 
			return i;
		}
	}
	return -1;
}

static int m_sort(const void *x, const void *y) 
{
	return (*(int*)x - *(int*)y);
}

#define  MAX_TAGS ( 1024 * 64)

// save all tags into the output file
bool CDICOMFile::SaveTags(int tag, FILE* outfile)
{
	bool retval=false;
	int tags[MAX_TAGS];
	int count=0, i;
	uint32_t total=0;
	bool tag_added=false;

	// check params
	if(outfile==NULL) 
		return retval;
	// get list of tags
	count=FindTags(tag, tags, &total, m_items);
	if(count==0) 
		return true;
	switch (tag)
	{
		case 0x7fe00010:
		case 0xFFFEE000:
		case 0xFFFEE0DD: break; // image data
		default: tag_added = AddDCMItem(tag, &total, m_items);
	}
	// sort array
	qsort((void*)tags, count, sizeof(int), m_sort);
	// Save tags
	if(tag_added) 
		retval=SaveTag(tag, outfile);
	else 
		retval=true;
	if (tag == 0xFFFEE000)
	{
		i = FindItemIdx(tag);
		for (; i < m_items.size(); i++)
			SaveTag(i, outfile, m_items);
	}
	else
	{
		for (i = 0; retval && (i < count); i++)
		{
			retval = SaveTag(tags[i], outfile);
		}
	}
	return retval;
}

int CDICOMFile::FindTags(int tag, int* tags, uint32_t* total_len, tag_list& _items)
{
	int retval=0, i=0;
	DCM_TAG xtag;

	*total_len=0;
	xtag.tag=tag;
	for(i; i<_items.size(); i++) 
	{	
		if(_items[i].tag.val[1]==xtag.val[1]) 
		{
			if (_items[i].size!=-1)
				*total_len+=_items[i].size;
			*total_len+=8; // TAG+VR+LEN
			if(IsBigData(_items[i].vr)) 
				*total_len+=4;
			*( tags + retval) = _items[i].tag.tag;
			retval++;
		}
	}
	return retval;
}

uint32_t sw(uint32_t d);

bool CDICOMFile::SaveTag(int tag, FILE* outfile)
{
	bool retval = false;
	int idx;

	if (outfile == NULL)
		return retval;
	idx = FindItemIdx(tag);
	if (idx < 0)
		return retval;
	else 
		return SaveTag(idx, outfile, m_items);
}

bool CDICOMFile::SaveTag(int idx, FILE* outfile, tag_list& _items)
{
	bool retval=false;
	uint16_t len;
	int tag;

	if(outfile==NULL) 
		return retval;
	if(idx<0) 
		return retval;
	if ( idx>=_items.size())
		return retval;
	tag = _items[idx].tag.tag;
	// write tag
	fwrite(&_items[idx].tag.val[1], sizeof(uint16_t), 1, outfile);
	fwrite(&_items[idx].tag.val[0], sizeof(uint16_t), 1, outfile);
	// write VR
	if (tag != 0xFFFEE000 && tag != 0xFFFEE0DD)
		fwrite(&_items[idx].vr, sizeof(uint16_t), 1, outfile);
	// write len
	if(IsBigData(_items[idx].vr))
	{
		len=0;
		if (tag != 0xFFFEE000 && tag != 0xFFFEE0DD)
			fwrite(&len, sizeof(uint16_t), 1, outfile);
		fwrite(&_items[idx].size, sizeof(uint32_t), 1, outfile);
	}
	else 
	{
		len=(uint16_t)_items[idx].size;
		fwrite(&len, sizeof(uint16_t), 1, outfile);
	}
	// write data
	if (_items[idx].data_ptr && _items[idx].size!=-1)
		fwrite(_items[idx].data_ptr, _items[idx].size, 1, outfile);
	return true;
}
bool CDICOMFile::IsBigData(uint16_t vr)
{
	bool retval=false;

	switch(vr)
	{
		case VR_OB :
		case VR_OW :
		case VR_SQ :
		case VR_UN :	retval=true; break;
		default : retval=false;
	}
	return retval;
}

int8_t*  CDICOMFile::GetImagePtr( int image_id)
{
	int8_t* retval=NULL;
	int cnt=0;

	cnt=GetImagesCount();
	if(cnt==0) 
		return retval;
	if(image_id<0) 
		return retval;
	if(image_id>=cnt) 
		return retval;
	return (int8_t*)v_Image_pointers.at(image_id);
}

uint64_t CDICOMFile::GetImageLength( int image_id)
{
	int retval=0;
	int cnt=0;

	cnt=GetImagesCount();
	if(cnt==0) 
		return retval;
	if(image_id<0) 
		return retval;
	if(image_id>=cnt) 
		return retval;
	return v_Image_lengths.at(image_id);
}



