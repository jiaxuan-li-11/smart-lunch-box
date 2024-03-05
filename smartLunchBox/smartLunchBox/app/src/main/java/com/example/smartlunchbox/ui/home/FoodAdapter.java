package com.example.smartlunchbox.ui.home;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.ViewGroup;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import com.example.smartlunchbox.databinding.FoodCardViewBinding;

import java.util.ArrayList;

public class FoodAdapter extends RecyclerView.Adapter<FoodAdapter.FoodViewHolder>{

    ArrayList<String> foodNames;
    ArrayList<Double> foodWeights;
    ArrayList<Double> foodCalories;
    Context context;

    public FoodAdapter(Context context,
                       ArrayList<String> foodNames,
                       ArrayList<Double> foodWeights,
                       ArrayList<Double> foodCalories){

        this.context = context;
        this.foodNames = foodNames;
        this.foodWeights = foodWeights;
        this.foodCalories = foodCalories;
    }

    @NonNull
    @Override
    public FoodViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        FoodCardViewBinding binding = FoodCardViewBinding.inflate(LayoutInflater.from(parent.getContext()));
        FoodViewHolder viewHolder = new FoodViewHolder(binding);
        return viewHolder;
    }

    @Override
    public void onBindViewHolder(@NonNull FoodViewHolder holder, int position) {
        holder.name.setText(foodNames.get(position));
        holder.weight.setText(String.format("%.2f", foodWeights.get(position))+" g");
        holder.calories.setText(String.format("%.2f", foodCalories.get(position))+" KCal");

    }

    @Override
    public int getItemCount() {
        return foodNames.size();
    }

    public class FoodViewHolder extends RecyclerView.ViewHolder {
        TextView name, weight, calories;
        public FoodViewHolder(FoodCardViewBinding binding){
            super(binding.getRoot());
            name = binding.getFoodName;
            weight = binding.getFoodWeight;
            calories = binding.getFoodCalorie;
        }
    }

    public void delete(){
        foodNames.clear();
        foodWeights.clear();
        foodCalories.clear();
        notifyDataSetChanged();
    }




}
