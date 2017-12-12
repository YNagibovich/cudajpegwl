/*
	u-pacs DICOM library
	Author : Y.Nagibovich
*/

#include "dcm_version.h"
#include "dcm_net.h"
#include "dcm_utils.h"
#include "dcm_client.h"

#include <string.h>
#include <stdio.h>
#include <direct.h>

typedef struct _get_data_ctx
{	
	dicom_session_context_t* session;
	char* buffer;
	bool retval;
	int total;
	dcm_logger_cb logger;
} echo_ctx_t;

DWORD WINAPI EchoProc(PVOID pParameter)
{	
	struct sockaddr_in addr;
	echo_ctx_t* get_ctx=(echo_ctx_t*) pParameter;
	if(get_ctx==NULL) 
		return 0;
	bool retval;

	// process
	// connect
	get_ctx->session->socket=socket(AF_INET, SOCK_STREAM,IPPROTO_TCP);
	if(get_ctx->session->socket==INVALID_SOCKET)
	{	
		//AfxMessageBox("Невозможно создать сокет отправителя.");
		if(get_ctx->session->event_exit) 
			SetEvent(get_ctx->session->event_exit);
		return 0;
	}
	addr.sin_family=AF_INET;
	addr.sin_port=htons((unsigned short)get_ctx->session->remote_port);
	addr.sin_addr.s_addr=get_inet_addr(get_ctx->session->remote_addr);
	if(connect(get_ctx->session->socket,(sockaddr*)&addr,sizeof(sockaddr))==SOCKET_ERROR)
	{	
		//AfxMessageBox("Ошибка соединения.",MB_ICONERROR);
		if(get_ctx->session->event_exit) 
			SetEvent(get_ctx->session->event_exit);
		return 0;
	}
	if(CreateAssociacion(get_ctx->buffer,get_ctx->session,ECHO_SOP,0)) 
	{	
		retval=DCM_CMD_ECHO_SCU(get_ctx->buffer,get_ctx->session);
		//m_lock.Lock();
		get_ctx->retval=retval;
		//m_lock.Unlock();
	}
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


bool DICOMEchoe(HWND parent, std::string sAET, DICOM_PROVIDER_t* info, dcm_logger_cb cb_logger)
{
	char buffer[MAX_BUFFER_SIZE];
	bool retval = false;
	DWORD dwThreadId;
	int ret_wait;
	dicom_session_context_t ctx;
	echo_ctx_t get_ctx;

	memset(&ctx,0,sizeof(dicom_session_context_t));
	// prepare context
	ctx.remote_addr=(char*)info->addr.c_str();
	ctx.remote_port=info->port;
	ctx.remote_aet=(char*)info->AET.c_str();
	ctx.logger = cb_logger;
	strncpy(ctx.assotiation_calling_AET, sAET.c_str(),16);
	// init progress 
	if( parent)
	{
		// PostMessage
		::PostMessage(parent,WM_USER_DCIOM_ECHO_START,0,0);
	}
	// infill context
	get_ctx.buffer=buffer;
	get_ctx.session=&ctx;
	get_ctx.retval=FALSE;
	get_ctx.total=10; // TBD
	get_ctx.logger = cb_logger;
	// prepare handlers
	ctx.event_exit=CreateEvent(NULL,FALSE,FALSE,NULL);
	// start thread
	CreateThread(NULL,0,EchoProc,&get_ctx,0,&dwThreadId);
	while(1)
	{	
		ret_wait=WaitForSingleObject(ctx.event_exit,ECHO_DEFAULT_TIMEOUT);
		retval=get_ctx.retval;
		if(ret_wait==WAIT_OBJECT_0) 
			break; // thread exit
		else 
		{	
			if( parent)
			{
				//step
				::PostMessage(parent, WM_USER_DCIOM_ECHO_STEP, 0, 0);
				retval = true;
			}
		}
	}
	if(parent) 
	{	
		// close
		::PostMessage(parent, WM_USER_DCIOM_ECHO_STOP, 0, 0);
	}
	CloseHandle(ctx.event_exit);
	return retval;
}
