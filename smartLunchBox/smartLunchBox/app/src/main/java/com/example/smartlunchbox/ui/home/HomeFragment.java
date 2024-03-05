package com.example.smartlunchbox.ui.home;

import android.os.Bundle;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.fragment.app.Fragment;
import androidx.lifecycle.ViewModelProvider;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.example.smartlunchbox.databinding.FragmentHomeBinding;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Timer;
import java.util.TimerTask;

import okhttp3.Request;

public class HomeFragment extends Fragment {

    private FragmentHomeBinding binding;
    private FoodAdapter foodAdapter;
    private boolean isDelete = false;
    private static final String urlWeight = "http://3.86.239.158:8080/meal";

    public View onCreateView(@NonNull LayoutInflater inflater,
                             ViewGroup container, Bundle savedInstanceState) {
        HomeViewModel homeViewModel =
                new ViewModelProvider(this).get(HomeViewModel.class);

        binding = FragmentHomeBinding.inflate(inflater, container, false);
        View root = binding.getRoot();

        setOkHttpResponse(container);

        final Button deleteButton = binding.buttonDeleteHome;
        deleteButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                foodAdapter.delete();
                binding.totalCal.setText("0 KCal");
                isDelete = true;
            }
        });

        final Button updateButton = binding.buttonUpdateHome;
        updateButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                setOkHttpResponse(container);
                isDelete = false;
            }
        });

        // refresh UI 8s
        if (isDelete == false){
            Timer timer = new Timer();
            timer.schedule(new TimerTask(){
                @Override
                public void run() {
                    onResume();
                    Log.d("refresh", "refresh UI");
                }
            }, 0, 10000);
        }


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

    public void setOkHttpResponse(ViewGroup container) {

        OkHttpResponse.getAsync(urlWeight, new OkHttpResponse.DataCallBack() {

            ArrayList<String> foodNames = new ArrayList<>();
            ArrayList<Double> foodWeights = new ArrayList<>();
            ArrayList<Double> foodCalories = new ArrayList<>();

            @Override
            public void requestFailure(Request request, IOException e) { }

            @Override
            public void requestSuccess(String getResponse) throws Exception {
                try {
                    JSONObject jsonObject = new JSONObject(getResponse);
                    JSONArray foodArray = jsonObject.getJSONArray("foods");
                    for (int i = 0; i < foodArray.length(); i++){
                        JSONObject item = foodArray.getJSONObject(i);
                        String name = item.getString("name");
                        foodNames.add(name);
                        Double weight = item.getDouble("weight");
                        foodWeights.add(weight);
                        Double calories = item.getDouble("calories");
                        foodCalories.add(calories);
                    }
                } catch (JSONException e) {
                    e.printStackTrace();
                }
                setRecyclerView(container, foodNames, foodWeights, foodCalories);
                setTotalCalView(foodCalories);

            }
        });

    }

    public void setTotalCalView(ArrayList<Double> foodCalories){
        final TextView totalCal = binding.totalCal;
        Double totalCalories = 0.0;
        for (int i = 0; i < foodCalories.size(); i++){
            totalCalories += foodCalories.get(i);
        }
        String showCalories = String.format("%.2f", totalCalories)+" KCal";
        totalCal.setText(showCalories);
    }

    public void setRecyclerView(ViewGroup container,
                                ArrayList<String> foodNames,
                                ArrayList<Double> foodWeights,
                                ArrayList<Double> foodCalories){

        final RecyclerView recyclerView = binding.foodRecycleView;
        LinearLayoutManager linearLayoutManager = new LinearLayoutManager(getContext());
        recyclerView.setLayoutManager(linearLayoutManager);

        foodAdapter = new FoodAdapter(container.getContext(), foodNames, foodWeights, foodCalories);
        recyclerView.setAdapter(foodAdapter);
    }




}