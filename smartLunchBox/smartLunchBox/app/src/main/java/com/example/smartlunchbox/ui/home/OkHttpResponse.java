package com.example.smartlunchbox.ui.home;

import android.os.Handler;
import android.os.Looper;

import androidx.annotation.NonNull;

import java.io.IOException;

import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.Response;


public class OkHttpResponse {
    private static OkHttpResponse okHttpManager;
    private static OkHttpClient client;
    private final Handler handler;

    private OkHttpResponse(){
        client = new OkHttpClient();
        handler = new Handler(Looper.getMainLooper());
    }

    public static OkHttpResponse getInstance(){
        if (okHttpManager == null){
            okHttpManager = new OkHttpResponse();
        }
        return okHttpManager;
    }

    public static void getAsync(String url, DataCallBack callBack){
        getInstance().asyncGET(url, callBack);
    }

    public void asyncGET(String url, DataCallBack callBack) {
        final Request request = new Request.Builder().url(url).build();
        Call call = client.newCall(request);
        call.enqueue(new Callback() {
            @Override
            public void onFailure(@NonNull Call call, @NonNull IOException e) {
                failureDelivery(request, e, callBack);
            }

            @Override
            public void onResponse(@NonNull Call call, @NonNull Response response) {
                String getResponse = null;
                try{
                    getResponse = response.body().string();
                }catch (IOException e){
                    failureDelivery(request, e, callBack);
                }
                successDelivery(getResponse, callBack);
            }
        });
    }


    private void failureDelivery(final Request request, final IOException e, final DataCallBack callBack) {
        handler.post(new Runnable() {
            @Override
            public void run() {
                if (callBack != null){
                    callBack.requestFailure(request, e);
                }
            }
        });
    }

    private void successDelivery(final String getResponse, final DataCallBack callBack) {
        handler.post(new Runnable() {
            @Override
            public void run() {
                if (callBack != null){
                    try{
                        callBack.requestSuccess(getResponse);
                    } catch (Exception e){
                        e.printStackTrace();
                    }
                }
            }
        });
    }

    public interface DataCallBack {
        void requestFailure (Request request, IOException e);
        void requestSuccess(String getResponse) throws Exception;
    }
}
