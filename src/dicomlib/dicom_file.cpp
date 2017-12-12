/*
	dicom_file.cpp
    u-pacs DICOM library

	Author : Y.Nagibovich

    ynagibovich@gmail.com
*/
#include "dicom_file.h"

#include <assert.h>

#define DICOM_FILE_TAG	"DICM"
#define COLOR_MONO      "MONOCHROME"

CDicomFile::CDicomFile(const char* pFilename /* = nullptr*/, uint16_t nStopGroup/* = 0*/, uint16_t nStopTag /* = 0*/):
    m_pRawData(nullptr)
{
    if (pFilename != nullptr && strlen(pFilename))
    {
        loadFile(pFilename, nStopGroup, nStopTag);
    }
}

CDicomFile::~CDicomFile(void)
{
    m_vImageLengths.clear();
    m_vTags.clear();
    m_vImagePointers.clear();
    
    if (m_pRawData != nullptr)
    {
        delete [] m_pRawData;
    }
}

bool CDicomFile::isBigData(uint16_t vr)
{
    bool retval = false;

    switch (vr)
    {
        case VR_OB:
        case VR_OW:
        case VR_SQ:
        case VR_UN:	retval = true; break;
        default: retval = false;
    }
    return retval;
}

bool CDicomFile::loadFile(const char* pFilename, uint16_t nStopGroup /*= 0*/, uint16_t nStopTag /*= 0*/)
{
    bool bRet = false;
    if (!m_sFilename.empty())
    {
        // already open
        return bRet;
    }

    m_sFilename = pFilename;

    std::ifstream stream( pFilename, std::ifstream::binary);
    if (!stream.good())
    {
        return bRet;
    }

    stream.seekg(0, std::ios::end);
    int nInputLength = (int)stream.tellg();
    stream.seekg(0, std::ios::beg);

    if (nInputLength < (128 + 4)) // too short
    {
        return bRet;
    }

    m_pRawData = new unsigned char[nInputLength];
    stream.read(reinterpret_cast<char *>(m_pRawData), nInputLength);

    unsigned char* ptr = (m_pRawData + 128);
    int nOffset = 0;
    if (!strncmp((char*)ptr, DICOM_FILE_TAG, 4))
    {
        nOffset = 128 + 4;
    }
    ptr = m_pRawData;
    m_vTags.clear();
    // parse data
    parseRawData(ptr, nOffset, nInputLength, nStopGroup, nStopTag);
    return isValidFile();
}

bool CDicomFile::isValidFile()
{
    return ((m_vTags.size() > 0) && ( m_vImagePointers.size() > 0));
}

// TBD add skip tag
bool CDicomFile::parseRawData(const unsigned char* pData, const int nOffset, const int nTotalSize, const uint16_t nStopGroup /*= 0*/, const uint16_t nStopTag /*= 0*/)
{	
	unsigned char* ptr = (unsigned char*)( pData + nOffset);
	bool bSkip = false;
	bool bDir = false;
	bool bAddImage = false;
	DATA_SET* ds = nullptr;
	DATA_SET_C* ds_r = nullptr;
	DATA_SET_B* ds_b = nullptr;
	DATA_SET_C* ds_c = nullptr;
	int nBytes, nTagSize = 0, mode = 0;
    uint32_t nLen = 0;
    bool bCheckStop = false;

    if (nStopGroup > 0 && nStopTag > 0)
    {
        bCheckStop = true;
    }
	
	for(nBytes = nOffset; nBytes < nTotalSize; )
	{	
        DCM_DATA tag;
        bAddImage = false;
    	ds=(DATA_SET*)ptr;

		if(!bDir)
        {	
            if (ds->vr<VR_DA || ds->vr>VR_OW)
            {
                mode = 1;
            }
		}
		if(!mode)	
		{	
			ds=(DATA_SET*)ptr;
            std::swap(ds->tag.val[0], ds->tag.val[1]);
            tag.tag = ds->tag;
		}
		else
		{	
			ds_r=(DATA_SET_C*)ptr;
            std::swap(ds_r->tag.val[0], ds_r->tag.val[1]);
            tag.tag = ds_r->tag;
		}

        // check stop condition
        if (bCheckStop)
        {
            if ((tag.tag.val[1] == nStopGroup) && (tag.tag.val[0] == nStopTag))
            {
                break;
            }
        }

		bSkip = false;
        tag.item_ptr = ptr;
		if(!mode)
		{	
			if ( ds)
			{
				switch(ds->tag.tag)
				{	
					case 0xFFFAFFFA :
					case 0xFFFEE000 : 
					case 0xFFFEE00D : 
					case 0xFFFEE0DD :	tag.vr=VR_SQ;
										ds_c=(DATA_SET_C*)ptr;
										nLen = ds_c->val_size;
                                        if (nLen == 0xFFFFFFFF)
                                        {
                                            nLen = 0;
                                        }
										if(!bDir)
										{	
                                            nTagSize = sizeof(DATA_SET_C) + nLen;
											bSkip = true;
										}
                                        else
                                        {
                                            nTagSize = sizeof(DATA_SET_C);
                                        }
										break; 
					default : break;
				}
			}
		}
		else
		{
            // TBD dir
		}
		if(!bSkip)
		{	
			if(!mode)
			{	
				switch(ds->vr)
				{	
					case VR_OB :
					case VR_OW :
					case VR_SQ :
					case VR_UN :    ds_b=(DATA_SET_B*)ptr;
		                            nLen = ds_b->val_size;
                                    if (nLen == 0xFFFFFFFF)
                                    {
                                        nLen = 0;
                                    }
                                    if (bDir)
                                    {
                                        nLen = 0;
                                    }
                                    nTagSize = sizeof(DATA_SET_B) + nLen;
					                break;
					default :	    nLen = ds->val_size;
                                    nTagSize =sizeof(DATA_SET) + nLen;
								    break;
				}
				tag.vr=(VR)ds->vr;
			}
			else
			{	
				ds_c=(DATA_SET_C*)ptr;
				nLen = ds_c->val_size;
                if (nLen == 0xFFFFFFFF)
                {
                    nLen = 0;
                }
                nTagSize = sizeof(DATA_SET_C) + nLen;
				//tag.vr=GetItemVR(ds_c->tag.tag);
                //assert(0);
                tag.vr = VR::VR_UN; // TBD
			}
		}
		tag.size=nLen;
		tag.data_ptr=(ptr+ nTagSize - nLen);
		if(!mode)
		{	
			if(ds->tag.tag==0x7FE00010 || ds->tag.tag == 0xFFFEE000)
			{	
				bAddImage = true;
			}
		}
		else
		{	
			if (ds_r)
			{
				if (ds_r->tag.tag == 0x7FE00010 || ds_r->tag.tag == 0xFFFEE000)
				{
					bAddImage = true;
				}
			}
		}
		if(bAddImage)
		{
            m_vImagePointers.push_back(tag.data_ptr);
            m_vImageLengths.push_back(tag.size);
		}
        m_vTags.push_back(tag);
        // next step
		ptr += nTagSize;
		nBytes += nTagSize;
	}
	return true;
}

bool CDicomFile::getWL(WL_t* wlt)
{	
	bool bRet = checkWL();

    if( bRet)
    {
        wlt->lut=0;
        int nVal = 65535; // def
        if (!getValue( 0x0028, 0x0120, nVal))
        {
            // TBD add2log
        }
        wlt->padding = (uint16_t)nVal;
        
        if (!getValue(0x0028, 0x1050, nVal))
        {
            // TBD add2log
            wlt->center = 0;
            wlt->lut = 1;
        }
        else
        {
            wlt->center = nVal;
        }

        if (!getValue(0x0028, 0x1051, nVal))
        {
            // TBD add2log
            wlt->intercept = 0;
        }
        else
        {
            wlt->intercept = nVal;
        }

        if (!getValue(0x0028, 0x1052, nVal))
        {
            // TBD add2log
            wlt->width = 1;
        }
        else
        {
            wlt->width = nVal;
        }

        if (!getValue(0x0028, 0x1053, nVal))
        {
            // TBD add2log
            wlt->slope = 1;
        }
        else
        {
            wlt->slope = nVal;
            if (wlt->slope == 0)
            {
                wlt->slope = 1;
            }
        }
	}
	return bRet;
}

bool CDicomFile::checkWL()
{	
	bool bRet=false;
	
    DCM_DATA data;
    if( getTag(0x0028, 0x0004, data))
	{	
        char* param = (char*)data.data_ptr;

        if (param && !_strnicmp(param, COLOR_MONO, strlen(COLOR_MONO)))
        {
            bRet = true;
        }
	}
	return bRet;
}

size_t CDicomFile::getImagesCount()
{
    return m_vImagePointers.size();
}

unsigned char* CDicomFile::getImageData(size_t& nImageSize, size_t nImageIdx/* = 0*/)
{
    if (nImageIdx < m_vImagePointers.size())
    {
        nImageSize = m_vImageLengths[nImageIdx];
        return m_vImagePointers[nImageIdx];
    }
    else
    {
        return nullptr;
    }
}

bool CDicomFile::getTag(DCM_DATA& data)
{
    bool bRet = false;
    
    for (auto it = m_vTags.begin(); it != m_vTags.end(); ++it)
    {
        if (data.tag.tag == it->tag.tag)
        {
            data = *it;
            bRet = true;
            break;
        }
    }
    return bRet;
}

bool CDicomFile::getTag(const uint16_t nGroup, const uint16_t nTag, DCM_DATA& data)
{
    data.tag.val[1] = nGroup;
    data.tag.val[0] = nTag;
    return getTag(data);
}

bool CDicomFile::getValue(const uint16_t nGroup, const uint16_t nTag, int& nValue)
{
    DCM_DATA data;
    
    bool bRet = getTag(nGroup, nTag, data);
    {
        if (bRet)
        {
            if (data.vr == VR_US)
            {
                uint16_t val = *(uint16_t*)data.data_ptr;
                nValue = val;
            }
            else if (data.vr == VR_SS)
            {
                int16_t val = *(int16_t*)data.data_ptr;
                nValue = val;
            }
            else if (data.vr == VR_UL)
            {
                uint32_t val = *(uint32_t*)data.data_ptr;
                nValue = val;
            }
            else if (data.vr == VR_SL)
            {
                int32_t val = *(int32_t*)data.data_ptr;
                nValue = val;
            }
            else if (data.vr == VR_DS || data.vr == VR_IS)
            {
                std::string val;
                val.append((const char*)data.data_ptr, (size_t)data.size);
                nValue = std::stoi(val);
            }
            else if (data.vr == VR_UN)
            {
                if (data.size == 2) // assume US
                {
                    uint16_t val = *(uint16_t*)data.data_ptr;
                    nValue = val;
                }
                else if (data.size == 4) // assume UL
                {
                    uint32_t val = *(uint32_t*)data.data_ptr;
                    nValue = val;
                }
                else
                {
                    bRet = false;
                }
            }
            else
            {
                bRet = false;
            }
        }
    }
    return bRet;
}

bool CDicomFile::getValue(const uint16_t nGroup, const uint16_t nTag, std::string& sValue)
{
    DCM_DATA data;

    bool bRet = getTag(nGroup, nTag, data);
    {
        if (bRet)
        {
            switch (data.vr)
            {
                case VR_UN : // shoot into the leg here :)
                case VR_AE :
                case VR_AS :
                case VR_AT :
                case VR_CS :
                case VR_DA :
                case VR_DS :
                case VR_DT :
                case VR_IS :
                case VR_LO :
                case VR_LT :
                case VR_OB :
                case VR_OW :
                case VR_PN :
                case VR_SH :
                case VR_ST :
                case VR_TM :
                case VR_UI :
                case VR_UT : sValue.clear(); sValue.append((const char*)data.data_ptr, (size_t)data.size); break;
                default:  bRet = false; break;
            }
        }
    }
    return bRet;
}

