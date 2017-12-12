/*
	u-pacs DICOM library
	Author : Y.Nagibovich
*/

#include "dcm_version.h"
#include "dcm_net.h"
#include "dcm_utils.h"
#include "dcm_client.h"
#include "dcm_file.h"

#include <string.h>
#include <stdio.h>
#include <direct.h>

typedef struct _send_data_ctx
{	
	dicom_session_context_t* session;
	char* buffer;
	bool retval;
	int total;
	std::string filename;
	std::string p_context;
	dcm_logger_cb logger;
} send_ctx_t;

#define SEND_DEFAULT_TIMEOUT	500 // 0.5 sec

//////////////////////////////////////////////////////////////////////////
// sender

DWORD WINAPI SendProc(PVOID pParameter)
{	
	struct sockaddr_in addr;
	send_ctx_t* get_ctx=(send_ctx_t*) pParameter;
	int jcnt=0;
	image_store_info_t info;
	std::string tempfile;
	CDICOMFile dcm;
	//CString delfile;

	if(get_ctx==NULL) return 0;
	get_ctx->session->socket=socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
	if(get_ctx->session->socket==INVALID_SOCKET)
	{	
		//AfxMessageBox("Невозможно создать сокет отправителя.");
		if(get_ctx->session->event_exit) 
			SetEvent(get_ctx->session->event_exit);
		get_ctx->retval=FALSE;
		return 0;
	}
	addr.sin_family=AF_INET;
	addr.sin_port=htons((unsigned short)get_ctx->session->remote_port);
	addr.sin_addr.s_addr=get_inet_addr(get_ctx->session->remote_addr);
	if(connect(get_ctx->session->socket, (sockaddr*)&addr, sizeof(sockaddr))==SOCKET_ERROR)
	{	
		//AfxMessageBox("Ошибка соединения.",MB_ICONERROR);
		if(get_ctx->session->event_exit) 
			SetEvent(get_ctx->session->event_exit);
		get_ctx->retval=FALSE;
		return 0;
	}
	if (CreateAssociacion(get_ctx->buffer, get_ctx->session, (char*)get_ctx->p_context.c_str(), 4)) // 1.2.840.10008.1.2.4.50
	{
		dcm.Init();
		if (dcm.OpenDICOMFile((char*)get_ctx->filename.c_str()))
		{
			// create storage
			tempfile = GetTempFile();
			info.datafile = dcm.Convert2SendEx((char*)tempfile.c_str());
			info.image_UID = dcm.GetDCMItemEx(0x00080018);
			info.sop_UID = dcm.GetDCMItemEx(0x00080016);
			//get_ctx->logger(0, L_INFO, "STORE SCU start : SOP UID = %s, image UID  = %s, Assoc = %d:%d, TS = %s",
			//	info.sop_UID.c_str(), info.image_UID.c_str(), get_ctx->session->assotiation_mode, 
			//	get_ctx->session->association_PID, get_ctx->p_context.c_str());
			get_ctx->retval = DCM_CMD_STORE_SCU(get_ctx->buffer, get_ctx->session, &info, 0); // simple mode
			//if (!get_ctx->retval)
			//	get_ctx->logger(0, L_ERROR, "STORE SCU failed");
			
			// TBD
			std::wstring tempfile_w = s2ws(tempfile);
			DeleteFile(tempfile_w.c_str());
		}
		//else
		//	get_ctx->logger(0, L_ERROR, "Cannot open DICOM file");
	}
	//else
	//	get_ctx->logger(0, L_ERROR, "Cannot create association. Error - %d", get_ctx->session->err_code);
	if(get_ctx->session->end_ass_rq)
	{	
		DCMEndAssociationRply(get_ctx->session,get_ctx->buffer);
	}
	else
	{	
		DCMEndAssociationRq(get_ctx->session,get_ctx->buffer);
		DCMEndAssociationGetRply(get_ctx->session,get_ctx->buffer);
	}
	//close connection
	if(get_ctx->session)
	{	
		shutdown(get_ctx->session->socket,SD_BOTH);
		closesocket(get_ctx->session->socket);
		if(get_ctx->session->event_exit) 
			SetEvent(get_ctx->session->event_exit);
	}
	return 1;

}

bool DICOMSend(HWND parent, std::string sAET, DICOM_PROVIDER_t* info, const char* file_name, dcm_logger_cb cb_logger)
{

	char buffer[MAX_BUFFER_SIZE];
	bool retval = false;
	DWORD dwThreadId;
	int ret_wait;
	dicom_session_context_t ctx;
	send_ctx_t send_ctx;

	memset(&ctx, 0, sizeof(dicom_session_context_t));
	// prepare context
	ctx.remote_addr = (char*)info->addr.c_str();
	ctx.remote_port = info->port;
	ctx.remote_aet = (char*)info->AET.c_str();
	strncpy(ctx.assotiation_calling_AET, sAET.c_str(), 16);
	ctx.logger = cb_logger;
	// init progress 
	if (parent)
	{
		PostMessage(parent, WM_USER_DCIOM_SEND_START, 0, 0);
	}
	// infill context
	send_ctx.buffer = buffer;
	send_ctx.session = &ctx;
	send_ctx.retval = false;
	send_ctx.total = 10;
	send_ctx.filename = file_name;
	send_ctx.logger = cb_logger;

	// TBD
#ifdef USE_WSI
	send_ctx.p_context = SC_WSI_IMAGE_STORAGE;
#else
	send_ctx.p_context = SC_CAPTURE_IMAGE_STORAGE;
#endif // USE_WSI
	ctx.err_code = 0;
	// prepare handlers
	ctx.event_exit = CreateEvent(NULL, FALSE, FALSE, NULL);
	// start thread
	CreateThread(NULL, 0, SendProc, &send_ctx, 0, &dwThreadId);
	while (1)
	{
		ret_wait = WaitForSingleObject(ctx.event_exit, SEND_DEFAULT_TIMEOUT);
		retval = send_ctx.retval;
		if (ret_wait == WAIT_OBJECT_0)
			break; // thread exit
		else
		{
			if (parent)
			{
				//step
				PostMessage(parent, WM_USER_DCIOM_SEND_STEP, 0, 0);
			}
		}
	}
	if (parent)
	{
		// close
		PostMessage(parent, WM_USER_DCIOM_SEND_STOP, 0, 0);
	}
	CloseHandle(ctx.event_exit);
	return false;
}





