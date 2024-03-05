package com.example.smartlunchbox.ui.dashboard;

import static android.content.Context.MODE_PRIVATE;

import android.app.Dialog;
import android.content.Context;
import android.content.SharedPreferences;
import android.graphics.Color;
import android.graphics.drawable.ColorDrawable;
import android.os.Bundle;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.fragment.app.Fragment;
import androidx.lifecycle.ViewModelProvider;

import com.example.smartlunchbox.databinding.DialogSetTargetCalBinding;
import com.example.smartlunchbox.databinding.FragmentDashboardBinding;
import com.example.smartlunchbox.ui.home.OkHttpResponse;
import com.github.mikephil.charting.charts.BarChart;
import com.github.mikephil.charting.charts.PieChart;

import org.json.JSONObject;

import java.io.IOException;
import java.util.ArrayList;

import okhttp3.Request;

public class DashboardFragment extends Fragment {

    private FragmentDashboardBinding binding;
    private DialogSetTargetCalBinding dialogBinding;
    private Dialog dialog;
    float usedCal = 0;
    private String urlDay = "http://3.86.239.158:8080/caltoday";
    private String urlWeek = "http://3.86.239.158:8080/calweek";

    public View onCreateView(@NonNull LayoutInflater inflater,
                             ViewGroup container, Bundle savedInstanceState) {
        DashboardViewModel dashboardViewModel =
                new ViewModelProvider(this).get(DashboardViewModel.class);
        binding = FragmentDashboardBinding.inflate(inflater, container, false);
        View root = binding.getRoot();

        setDialogView(container);
        setOkHttpPieChartView(container);
        setOkHttpBarChartView();
        setPieChartView(usedCal, container);

        return root;
    }


    @Override
    public void onDestroyView() {
        super.onDestroyView();
        binding = null;
    }

    @Override
    public void onResume() {
        super.onResume();
        this.onCreate(null);
    }


    public void setDialogView(ViewGroup container){
        Button setTargetButton = binding.setButton;
        setTargetButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                dialogBinding = DialogSetTargetCalBinding.inflate(getLayoutInflater());
                dialog = new Dialog(container.getContext());
                dialog.setContentView(dialogBinding.getRoot());
                dialog.getWindow().setBackgroundDrawable(new ColorDrawable(Color.WHITE));
                dialog.getWindow().setLayout(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);

                Button setButton = dialogBinding.setButton;
                setButton.setOnClickListener(new View.OnClickListener() {
                    @Override
                    public void onClick(View view) {
                        String targetCal = dialogBinding.targetCal.getText().toString();
                        if(!targetCal.isEmpty()){
                            try {
                                float targetDayCal = Float.parseFloat(targetCal.toString());
                                Log.d("targetCal in on click", targetCal.toString());
                                setTargetDayCal(targetDayCal, container.getContext());
                                setPieChartView(usedCal, container);
                                setOkHttpPieChartView(container);
                            } catch (NumberFormatException nfe) {
                                Toast.makeText(container.getContext(), "Enter A Valid Value", Toast.LENGTH_SHORT).show();
                            }

                        }
                        else{
                            Toast.makeText(container.getContext(), "Enter A Valid Value", Toast.LENGTH_SHORT).show();
                        }
                        dialog.dismiss();
                    }
                });
                dialog.show();

            }
        });
    }

    public void setOkHttpPieChartView(ViewGroup container) {
        OkHttpResponse.getAsync(urlDay, new OkHttpResponse.DataCallBack() {
            @Override
            public void requestFailure(Request request, IOException e) { }

            @Override
            public void requestSuccess(String getResponse) throws Exception {
                usedCal = Float.parseFloat(getResponse);
                setPieChartView(usedCal, container);
            }

        });
    }

    public void setOkHttpBarChartView(){
        OkHttpResponse.getAsync(urlWeek, new OkHttpResponse.DataCallBack() {
            @Override
            public void requestFailure(Request request, IOException e) { }

            @Override
            public void requestSuccess(String getResponse) throws Exception {
                JSONObject jsonObject = new JSONObject(getResponse);
                ArrayList<Float> calWeek = new ArrayList<Float>();
                for (int i = 1; i < 8; i++){
                     Double calD  = jsonObject.getDouble(String.valueOf(i));
                     float calF = calD.floatValue();
                     calWeek.add(calF);
                }
                Log.d("ResponseCalWeek", calWeek.toString());
                setBarChartView(calWeek);
            }

        });
    }

    public void setPieChartView(float usedCal, ViewGroup container){
        float dayLeft = 3000-usedCal;
        float targetDayCal = getTargetDayCal(container.getContext());
        if (targetDayCal > 0){
            dayLeft = targetDayCal - usedCal;
            if (dayLeft < 0){
                dayLeft = 0;
            }
        }
        Log.d("targetCal in pie chart", String.valueOf(targetDayCal));

        final PieChart myPieChart = binding.pieChart;
        DashboardViewModel dashboardViewModel =
                new ViewModelProvider(this).get(DashboardViewModel.class);
        dashboardViewModel.pieChart(myPieChart, usedCal, dayLeft);
    }

    public void setBarChartView(ArrayList<Float> barList){
        BarChart myBarChart = binding.barChart;
        DashboardViewModel dashboardViewModel =
                new ViewModelProvider(this).get(DashboardViewModel.class);
        dashboardViewModel.barChart(myBarChart, barList);
    }


    public static float getTargetDayCal(Context context){
        SharedPreferences sharedPreferences = context.getSharedPreferences("dayCal", MODE_PRIVATE);
        String calString = sharedPreferences.getString("targetCal", "");
        float cal;
        try{
            cal = Float.parseFloat(calString);
        }catch(Exception e){
            cal = 3000;
        }
        Log.d("get target cal", calString);
        return cal;
    }

    public void setTargetDayCal(float cal, Context context){
        SharedPreferences sharedPreferences = context.getSharedPreferences("dayCal", MODE_PRIVATE);
        SharedPreferences.Editor editor = sharedPreferences.edit();
        editor.putString("targetCal", String.valueOf(cal));
        editor.commit();
    }

}